"""
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
"""
from typing import List
from json import decoder
import warnings

warnings.filterwarnings("ignore")

from models.vit import VisionTransformer, interpolate_pos_embed
from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F

import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file


class BLIP_Base(nn.Module):
    def __init__(
        self,
        med_config="configs/med_config.json",
        image_size=224,
        vit="base",
        vit_grad_ckpt=False,
        vit_ckpt_layer=0,
    ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.visual_encoder, vision_width = create_vit(
            vit, image_size, vit_grad_ckpt, vit_ckpt_layer
        )
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)

    def forward(self, image, caption, mode):

        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode parameter must be image, text, or multimodal"
        text = self.tokenizer(caption, return_tensors="pt").to(image.device)

        if mode == "image":
            # return image features
            image_embeds = self.visual_encoder(image)
            return image_embeds

        elif mode == "text":
            # return text features
            text_output = self.text_encoder(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode="text",
            )
            return text_output.last_hidden_state

        elif mode == "multimodal":
            # return multimodel features
            image_embeds = self.visual_encoder(image)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            text.input_ids[:, 0] = self.tokenizer.enc_token_id
            output = self.text_encoder(
                text.input_ids,
                attention_mask=text.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            return output.last_hidden_state


def logits_to_ppl(logits, input_ids, input_attention_mask, prompt_length: int):
    output_ids = input_ids[:, prompt_length:]
    output_attention_mask = input_attention_mask[:, prompt_length:]
    probs = torch.log_softmax(logits, dim=-1)  # [B, L, N_vocab]
    probs = torch.gather(probs, dim=-1, index=output_ids.unsqueeze(-1))
    ppl = (probs.squeeze(-1) * output_attention_mask.float()).sum(dim=-1)

    ppl /= output_attention_mask.float().sum(dim=-1)

    return ppl


class BLIP_Decoder(nn.Module):
    def __init__(
        self,
        med_config="configs/med_config.json",
        image_size=384,
        vit="base",
        vit_grad_ckpt=False,
        vit_ckpt_layer=0,
        prompt="a picture of ",
    ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.visual_encoder, vision_width = create_vit(
            vit, image_size, vit_grad_ckpt, vit_ckpt_layer
        )
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_decoder = BertLMHeadModel(config=med_config)

        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

    def set_prompt(self, prompt):
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

    def forward(self, image, caption):

        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        text = self.tokenizer(
            caption,
            padding="longest",
            truncation=True,
            max_length=40,
            return_tensors="pt",
        ).to(image.device)

        text.input_ids[:, 0] = self.tokenizer.bos_token_id

        decoder_targets = text.input_ids.masked_fill(
            text.input_ids == self.tokenizer.pad_token_id, -100
        )
        decoder_targets[:, : self.prompt_length] = -100

        decoder_output = self.text_decoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            labels=decoder_targets,
            return_dict=True,
        )
        loss_lm = decoder_output.loss

        return loss_lm

    def encode_images(self, image):
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        return image_embeds, image_atts

    def perplexity(self, image_embeds, prompt: str, outputs: List[str]):
        bs = len(outputs)
        image_embeds = image_embeds.expand(bs, *image_embeds.size())
        # if isinstance(image, tuple):
        #     images_embeds, image_atts = image
        # else:
        #     image_embeds = self.visual_encoder(image)
        #
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image_embeds.device
        )
        # model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask":image_atts}

        tokenized_prompt = self.tokenizer.batch_encode_plus(
            [prompt], return_length=True
        )

        len_tokenized_prompt = tokenized_prompt.length[0] - 1

        prompts = [prompt] * bs
        prompts = [f"{prompt} {output}" for prompt, output in zip(prompts, outputs)]
        tokenized = self.tokenizer.batch_encode_plus(
            prompts, padding=True, return_tensors="pt", return_length=True
        ).to(image_embeds.device)
        attention_mask = tokenized.attention_mask
        input_ids = tokenized.input_ids
        input_ids[:, 0] = self.tokenizer.bos_token_id
        for idx in range(bs):
            # DELETE last [SEP] token
            attention_mask[idx, tokenized.length[idx].item() - 1] = 0

        # decoder_len = output_ids.shape[1]
        # ppl = torch.zeros([bs]).to(text.input_ids.device)
        outs = self.text_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            # return_logits=True,
        )
        logits = outs.logits  # [B, L, N_vocab]
        return logits_to_ppl(logits, input_ids, attention_mask, len_tokenized_prompt)

    def generate(
        self,
        image,
        sample=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
        num_return_sequences=1,
        length_penalty=1.0,
        prompts=None,
    ):
        image_embeds = self.visual_encoder(image)

        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }
        if prompts is None:
            prompt = [self.prompt] * image.size(0)
        else:
            prompt = prompts

        # temporarily use padding_side = left
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        input_ids = self.tokenizer(prompt, return_tensors="pt", padding="longest").input_ids.to(
            image.device
        )
        self.tokenizer.padding_side = old_padding_side
        # input_ids = self.tokenizer.batch_encode_plus(
        #     prompts, padding=：logest, return_tensors="pt", return_length=True
        # ).to(image.device)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]

        if sample:
            # nucleus sampling
            outputs = self.text_decoder.generate(
                input_ids=input_ids,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.1,
                length_penalty=length_penalty,
                **model_kwargs
            )
        else:
            # beam search
            outputs = self.text_decoder.generate(
                input_ids=input_ids,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=repetition_penalty,
                **model_kwargs
            )

        captions = []
        for i, output in enumerate(outputs):
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[len(prompt[i]) :])
        return captions


def blip_decoder(pretrained="", **kwargs):
    model = BLIP_Decoder(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        assert len(msg.missing_keys) == 0
    return model


def blip_feature_extractor(pretrained="", **kwargs):
    model = BLIP_Base(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        assert len(msg.missing_keys) == 0
    return model


def init_tokenizer():
    model_type = "bert-base-uncased"
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer_path = f"/home/mowentao/data/bert-vqa/saved_config/tokenizer/{model_type}"
    if os.path.exists(tokenizer_path):
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_type)
        tokenizer.save_pretrained(tokenizer_path)
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


def create_vit(
    vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0
):

    assert vit in ["base", "large"], "vit parameter must be base or large"
    if vit == "base":
        vision_width = 768
        visual_encoder = VisionTransformer(
            img_size=image_size,
            patch_size=16,
            embed_dim=vision_width,
            depth=12,
            num_heads=12,
            use_grad_checkpointing=use_grad_checkpointing,
            ckpt_layer=ckpt_layer,
            drop_path_rate=0 or drop_path_rate,
        )
    elif vit == "large":
        vision_width = 1024
        visual_encoder = VisionTransformer(
            img_size=image_size,
            patch_size=16,
            embed_dim=vision_width,
            depth=24,
            num_heads=16,
            use_grad_checkpointing=use_grad_checkpointing,
            ckpt_layer=ckpt_layer,
            drop_path_rate=0.1 or drop_path_rate,
        )
    return visual_encoder, vision_width


def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def load_checkpoint(model, url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(
            url_or_filename, check_hash=False, progress=True
        )
        checkpoint = torch.load(cached_file, map_location="cpu")
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location="cpu")
    else:
        raise RuntimeError("checkpoint url or path is invalid")

    state_dict = checkpoint["model"]

    state_dict["visual_encoder.pos_embed"] = interpolate_pos_embed(
        state_dict["visual_encoder.pos_embed"], model.visual_encoder
    )
    if "visual_encoder_m.pos_embed" in model.state_dict().keys():
        state_dict["visual_encoder_m.pos_embed"] = interpolate_pos_embed(
            state_dict["visual_encoder_m.pos_embed"], model.visual_encoder_m
        )
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape != model.state_dict()[key].shape:
                print("delete shape unmatched key in state_dict: ", key)
                del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)
    print("load checkpoint from %s" % url_or_filename)
    return model, msg


