from transformers.models.vilt.modeling_vilt import (
    ViltEmbeddings,
    ViltEncoder,
    ViltModel,
    ViltForQuestionAnswering,
    ViltLayer,
    ViltPooler,
)
from transformers import ViltProcessor
import torch
from torch import nn

import pretty_errors
from icecream import ic
from tqdm.auto import tqdm
from PIL import Image
import numpy as np

import logging
logger = logging.getLogger(__name__)
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPooling, SequenceClassifierOutput

class ViltEmbeddings3D(ViltEmbeddings):
    """
    Add 3D scene object embeddings to the input embeddings
    """

    def __init__(self, config):
        super().__init__(config)
    

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        pixel_values,
        pixel_mask,
        inputs_embeds,
        image_embeds,
        image_token_type_idx=1,
        scene_object_embeds=None,
        scene_object_mask=None,
    ):
        embeddings, masks = super().forward(
            input_ids,
            attention_mask,
            token_type_ids,
            pixel_values,
            pixel_mask,
            inputs_embeds,
            image_embeds,
            image_token_type_idx,
        )

        if scene_object_embeds is not None:
            # concat scene_object_embeds to embeddings
            embeddings = torch.cat([embeddings, scene_object_embeds], dim=1)
            masks = torch.cat([masks, scene_object_mask], dim=1)

        return embeddings, masks

class ViltEncoderTwin(ViltEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_hidden_layers_twin = getattr(config, 'num_hidden_layers_twin', config.num_hidden_layers)
        self.layer_twin = nn.ModuleList([ViltLayer(config) for _ in range(config.num_hidden_layers_twin)])

    def init_twin(self):
        logger.info('Twin-ViLT: Initializing twin encoder')
        # self.layer_twin.load_state_dict(self.layer.state_dict())
        for i in range(self.num_hidden_layers_twin):
            self.layer_twin[i].load_state_dict(self.layer[i].state_dict())


    def forward(self):
        ...


class ViltModelTwin(ViltModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.config = config

        self.encoder = ViltEncoderTwin(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViltPooler(config) if add_pooling_layer else None
        self.pooler_twin = ViltPooler(config) if add_pooling_layer else None

        self.post_init()
        self.init_twin()

    def init_twin(self):
        self.encoder.init_twin()
        if self.pooler is not None:
            self.pooler_twin.load_state_dict(self.pooler.state_dict())

    def forward(self):
        ...


class ViltModel3D(ViltModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer=add_pooling_layer)

        self.embeddings = ViltEmbeddings3D(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        image_token_type_idx: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        scene_object_embeds=None,
        scene_object_mask=None,
    ) -> Union[BaseModelOutputWithPooling, Tuple[torch.FloatTensor]]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import ViltProcessor, ViltModel
        >>> from PIL import Image
        >>> import requests

        >>> # prepare image and text
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "hello world"

        >>> processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        >>> model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")

        >>> inputs = processor(image, text, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        text_batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((text_batch_size, seq_length)), device=device)

        if pixel_values is not None and image_embeds is not None:
            raise ValueError("You cannot specify both pixel_values and image_embeds at the same time")
        elif pixel_values is None and image_embeds is None:
            raise ValueError("You have to specify either pixel_values or image_embeds")

        image_batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeds.shape[0]
        if image_batch_size != text_batch_size:
            raise ValueError("The text inputs and image inputs need to have the same batch size")
        if pixel_mask is None:
            pixel_mask = torch.ones((image_batch_size, self.config.image_size, self.config.image_size), device=device)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output, attention_mask = self.embeddings(
            input_ids,
            attention_mask,
            token_type_ids,
            pixel_values,
            pixel_mask,
            inputs_embeds,
            image_embeds,
            image_token_type_idx=image_token_type_idx,
            scene_object_embeds=scene_object_embeds,
            scene_object_mask=scene_object_mask,
        )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:] + (attention_mask,)

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class ViltVQA3D(ViltForQuestionAnswering):
    def __init__(self, config):
        super().__init__(config)
        self.vilt_base = self.vilt
        self.use_text_decoder = False
        # replace the VL backbone with a Twin-Transformer backbone
        if config.scene_feature_position == "twin":
            self.vilt = ViltModelTwin(config) # FIXME: ViLT does not support twin encoder
        elif config.scene_feature_position == "concat":
            self.vilt = ViltModel3D(config)
            self.vilt.load_state_dict(self.vilt_base.state_dict())
            del self.vilt_base
        else:
            raise NotImplementedError(f"scene_feature_position={config.scene_feature_position} not implemented")

        # self.tokenizer = vilt.ViltTokenizer.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.linear_scene_object = nn.Sequential(
            nn.Linear(config.scene_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size)
        )
        
    def forward(self, image, text,
        scene_object_embeds=None, scene_object_mask=None, embed_image=False, **kwargs):
        # transform numpy image to PIL image
        image = [Image.fromarray(img) for img in image.detach().cpu().numpy().astype(np.uint8)]

        # tokenize text
        encoding = self.processor(image, text, return_tensors="pt", padding='longest', truncation=True, max_length=80)
        for k, v in encoding.items():
            encoding[k] = v.to(scene_object_embeds.device)

        scene_object_embeds = self.linear_scene_object(scene_object_embeds) # to VLM hidden size

        outputs = self.vilt(
            scene_object_embeds=scene_object_embeds,
            scene_object_mask=scene_object_mask,
            return_dict=False,
            **encoding,
        )

        pooler_output = outputs[1]
        last_hidden_state = outputs[0]
        N_obj = scene_object_embeds.shape[1]
        # fused_object_features = last_hidden_state[:, -N_obj:, :] # take all 3D scene object features

        logits = self.classifier(pooler_output)

        # return logits, fused_object_features, scene_object_mask
        return logits, last_hidden_state, outputs[-1]




    