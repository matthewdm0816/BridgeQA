from models.med import BertConfig, BertModel, BertLMHeadModel, BertEncoder, BertPooler, BertLMPredictionHead, BertLMClassificationModel, BertPrefixModel, BertModelTwin
from models.blip import create_vit, init_tokenizer, load_checkpoint

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np
from copy import deepcopy
from models.vit import interpolate_pos_embed
import random
from argparse import Namespace
from icecream import ic
import os

SAVE_PATH = "./temp_model.pth"
DEFAULT_BLIP_CONFIG = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../BLIP/configs/med_config.json")

def to_all_answer_score(ans_idx, ans_score, num_answers, batch_size):
    all_answer_score = torch.zeros(
        [batch_size, len(num_answers)], device=ans_idx.device
    )
    for i in range(ans_score.size(0)):
        all_answer_score[i % batch_size][ans_idx[i]] += ans_score[i]
    all_answer_score = torch.where(
        all_answer_score == 0, -1e6, all_answer_score
    )
    return all_answer_score

@torch.no_grad()
def weight_reset(m: torch.nn.Module):
    # - check if the current module has reset_parameters & if it's callabed called it on m
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()

def concat_repeat(a, b, n_repeat):
    assert a.shape == b.shape
    ab = a.new_empty([a.size(0) + b.size(0), *a.shape[1:]])
    ab[0::2] = a
    ab[1::2] = b # [a1, b1, a2, b2, ...]
    ab = ab.repeat_interleave(n_repeat, dim=0) # [a1, a1, a1, b1, b1, b1, a2, a2, a2, b2, b2, b2, ...]
    return ab

class BLIP_VQA3D(nn.Module):
    def __init__(self,                 
                 #  med_config = '/home/mowentao/scratch/BLIP/configs/med_config.json',  
                 med_config = DEFAULT_BLIP_CONFIG,
                 image_size = 480,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 scene_size=128,           
                 num_answers=None,
                 use_text_decoder=False,
                 answer_pdrop=0.1,
                 scene_feature_position="paralleltwin",
                 use_scene_weight=False,
                 use_scene_classifier=False,
                 use_scene_classifier_2d3d=False,
                 not_copy_weights=False,
                 mix_tokens=False,
                 share_decoder=False,
                 num_hidden_layers_twin=None,
                 encoder_layers=None,
                 decoder_layers=None,
                 **kwargs,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        assert num_answers is not None, "num_answers must be specified"
        assert scene_feature_position == "paralleltwin"
        self.num_answers = num_answers
        self.use_text_decoder = use_text_decoder
        self.scene_feature_position = scene_feature_position

        self.use_scene_classifier = use_scene_classifier
        self.use_scene_classifier_2d3d = use_scene_classifier_2d3d
        self.not_copy_weights = not_copy_weights
        self.mix_tokens = mix_tokens
        self.share_decoder = share_decoder
        
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer, drop_path_rate=0.1)
        self.tokenizer = init_tokenizer()  

        print(f"Unused kwargs: {kwargs}")
        
        encoder_config = BertConfig.from_json_file(med_config)
        if encoder_layers is not None:
            encoder_config.num_hidden_layers = encoder_layers
        encoder_config.encoder_width = vision_width
        if num_hidden_layers_twin is not None:
            setattr(encoder_config, "num_hidden_layers_twin", num_hidden_layers_twin)

        # --- Twin Transformer
        self.text_encoder = BertModelTwin(config=encoder_config, add_pooling_layer=False)

        # --- Simple Fusion Module
        if self.parallel:
            lowrank = encoder_config.hidden_size // 8
            self.lowrank_2d = nn.Linear(encoder_config.hidden_size, lowrank)
            self.lowrank_3d = nn.Linear(encoder_config.hidden_size, lowrank)
            self.bilinear_fusion = nn.Bilinear(lowrank, lowrank, encoder_config.hidden_size)
            
        # --- Answer Decoder
        decoder_config = BertConfig.from_json_file(med_config)     
        if decoder_layers is not None:
            decoder_config.num_hidden_layers = decoder_layers
        self.text_decoder = BertLMHeadModel(config=decoder_config)

        # --- 3D Answer Decoder
        decoder_config_3d = deepcopy(decoder_config)
        if self.share_decoder:
            self.text_decoder_scene = self.text_decoder
        else:
            self.text_decoder_scene = BertLMHeadModel(config=decoder_config_3d)
    
        # --- Answer Classifier (if use_scene_classifier, then this is the final classifier)
        self.answer_cls = nn.Sequential(
            nn.Linear(encoder_config.hidden_size, encoder_config.hidden_size),
            nn.GELU(),
            nn.Dropout(answer_pdrop),
            nn.LayerNorm(encoder_config.hidden_size),
            nn.Linear(encoder_config.hidden_size, num_answers)
        )

        self.answer_cls_2d3d = nn.Sequential(
            nn.Linear(encoder_config.hidden_size, encoder_config.hidden_size),
            nn.GELU(),
            nn.Dropout(answer_pdrop),
            nn.LayerNorm(encoder_config.hidden_size),
            nn.Linear(encoder_config.hidden_size, num_answers)
        )

        # --- 3d adapter linear
        self.linear_scene_object = nn.Sequential(
            nn.Linear(scene_size, decoder_config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(decoder_config.hidden_size, decoder_config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(decoder_config.hidden_size)
        )

        # --- camera pose encoder
        self.camera_encoder = nn.Sequential(
            nn.Linear(16, decoder_config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(decoder_config.hidden_size, decoder_config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(decoder_config.hidden_size)
        )


        self.use_scene_weight = use_scene_weight
        self.scene_weight = nn.Parameter(torch.zeros(1, dtype=torch.float32) + 1e-5, requires_grad=True)

        self.pretrained = None # for later record 

        self.projection_head = nn.Sequential(
            nn.Linear(vision_width, vision_width),
            nn.GELU(),
            nn.LayerNorm(vision_width),
            nn.Dropout(0.1),
            nn.Linear(vision_width, 1),
            nn.Sigmoid(),
        )

        # tie weights of text encoder and text decoder (image/scene)
        self.copy_weights()

    @property
    def parallel(self):
        return self.scene_feature_position in ["parallel", "parallel++", "parallelshare", "paralleltwin"]

    def copy_weights(self):
        # copy encoder and decoder weights
        if self.parallel and not self.not_copy_weights:
            self.text_encoder.init_twin()
            if not self.share_decoder:
                self.text_decoder_scene.load_state_dict(self.text_decoder.state_dict(), strict=False)
            

    def save_state_dict(self):
        # self.saved_state_dict = deepcopy(self.state_dict())
        torch.save(self.state_dict(), SAVE_PATH)

    def reinit_params(self):
        # reload pretrained weights
        checkpoint = torch.load(self.pretrained, map_location="cpu")
        state_dict = checkpoint["model"]

        state_dict["visual_encoder.pos_embed"] = interpolate_pos_embed(
            state_dict["visual_encoder.pos_embed"], self.visual_encoder
        )
        if "visual_encoder_m.pos_embed" in self.state_dict().keys():
            state_dict["visual_encoder_m.pos_embed"] = interpolate_pos_embed(
                state_dict["visual_encoder_m.pos_embed"], self.visual_encoder_m
            )
        for key in self.state_dict().keys():
            if key in state_dict.keys():
                if state_dict[key].shape != self.state_dict()[key].shape:
                    print("delete shape unmatched key in state_dict: ", key)
                    del state_dict[key]


        msg = self.load_state_dict(state_dict, strict=False)
        print("reload checkpoint from %s" % self.pretrained)
        # self, msg = load_checkpoint(self, self.pretrained)

        # load text_encoder from checkpoint
        saved_state_dict = torch.load(SAVE_PATH, map_location=torch.device(f"cuda:{torch.distributed.get_rank()}"))
        filtered_state_dict = {k: v for k, v in saved_state_dict.items() if "text_encoder" in k}
        self.load_state_dict(filtered_state_dict, strict=False)

        self.copy_weights()



    def forward(self, image, question, answer=None, 
                n=None, weights=None, train=True, inference='rank', 
                k_test=128, image_embeds=None, 
                scene_object_embeds=None, scene_object_mask=None, image_pose=None, image_per_sample=1, embed_image=False, depth_map=None, data_dict=None):
        if image_embeds is None:
            image_embeds = self.visual_encoder(image) # [batch_size, num_patches, hidden_size]
        if embed_image:
            projection_weight = self.projection_head(image_embeds[:, 0]) # [batch_size, 1]

            return image_embeds, projection_weight
        
        if image_per_sample > 1:
            B, P, H = image_embeds.size()
            image_embeds = image_embeds.view(B//P, P*image_per_sample, H) # 
        

        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        question = self.tokenizer(question, padding='longest', truncation=True, max_length=80,  # original 35 
                                  return_tensors="pt").to(image.device) 
        question.input_ids[:,0] = self.tokenizer.enc_token_id
    
        # encoded_scene_object_embeds = None # FIXME: not implemented

        if self.use_scene_weight:
            print(f"scene weight: {self.scene_weight.data.item()}")
            scene_object_mask = scene_object_mask * torch.clamp(self.scene_weight, min=0, max=1)
            # scene_object_mask.fill_(0) # test: no scene object

        # concat scene object
        if scene_object_embeds is not None:
            print(image_embeds.shape, scene_object_embeds.shape)
            scene_object_embeds = self.linear_scene_object(scene_object_embeds) # [batch_size, num_objects, hidden_size]
            if image_pose is not None and not self.parallel:
                # concat camera pose into 3D scene object information
                image_pose = self.camera_encoder(image_pose) # [batch_size, hidden_size]
                scene_object_embeds = torch.cat([image_pose.unsqueeze(1), scene_object_embeds], dim=1)
                scene_object_mask = torch.cat([torch.ones(image_pose.size(0), 1, dtype=torch.long).to(image.device), scene_object_mask], dim=1)
            
        
        if self.scene_feature_position == "paralleltwin":
            question_output = self.text_encoder(question.input_ids, 
                                                attention_mask = question.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,      
                                                encoder_hidden_states_twin = scene_object_embeds,
                                                encoder_attention_mask_twin = scene_object_mask,
                                                return_dict = True,
                                                output_attentions=True,)
            question_attention_mask = question.attention_mask
            question_output_hidden_states, question_output_scene_hidden_states = question_output.last_hidden_state
            question_output.last_hidden_state = question_output_hidden_states
            question_output_scene = Namespace()
            question_output_scene.last_hidden_state = question_output_scene_hidden_states
            data_dict["2d_self_attention"], data_dict["3d_self_attention"] = question_output.attentions[-1] # take the last layer
            data_dict["2d_cross_attention"], data_dict["3d_cross_attention"] = question_output.cross_attentions[-1]
            ic(data_dict["2d_self_attention"].shape, data_dict["3d_self_attention"].shape, data_dict["2d_cross_attention"].shape, data_dict["3d_cross_attention"].shape)
            
        
        if train:               
            '''
            n: number of answers for each question
            weights: weight for each answer
            '''            
            if self.use_text_decoder:
                assert answer is not None, "answer must be specified if use text decoder (free-form answer mode)"
                answer = self.tokenizer(answer, padding='longest', return_tensors="pt").to(image.device) 
                answer.input_ids[:,0] = self.tokenizer.bos_token_id
                answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)      
                answer_output = self.text_decoder(answer.input_ids, 
                                              attention_mask = answer.attention_mask, 
                                              encoder_hidden_states = question_output.last_hidden_state,
                                              encoder_attention_mask = question_attention_mask,                  
                                              labels = answer_targets,
                                              return_dict = True,   
                                              reduction = 'none',
                                             )

                loss = answer_output.loss
                loss = loss.sum() / image_embeds.size(0)

                if self.scene_feature_position in ["parallel++", "parallelshare", "paralleltwin"]:
                    # predict from question_output_scene
                    if self.use_scene_classifier:
                        answer_score_scene = self.answer_cls(question_output_scene.last_hidden_state[:,0,:])

                        question_output.last_hidden_state = self.fuse_2d3d(question_output, question_output_scene)

                        if self.use_scene_classifier_2d3d:
                            answer_score_2d3d = self.answer_cls_2d3d(question_output.last_hidden_state[:,0,:])
                        else:
                            answer_score_2d3d = None
                
                        return (loss, answer_score_scene, answer_score_2d3d), question_output.last_hidden_state, question_attention_mask # or return answer_output.last_hidden_state?
                    else:
                        if self.scene_feature_position in ["parallel++", "paralleltwin"]:
                            answer_output_scene = self.text_decoder_scene(answer.input_ids,
                                                                            attention_mask = answer.attention_mask,
                                                                            encoder_hidden_states = question_output_scene.last_hidden_state,
                                                                            encoder_attention_mask = question_attention_mask,
                                                                            labels = answer_targets,
                                                                            return_dict = True,
                                                                            reduction = 'none',
                                                                        )
                        else: # parallelshare
                            answer_output_scene = self.text_decoder(answer.input_ids,
                                                                            attention_mask = answer.attention_mask,
                                                                            encoder_hidden_states = question_output_scene.last_hidden_state,
                                                                            encoder_attention_mask = question_attention_mask,
                                                                            labels = answer_targets,
                                                                            return_dict = True,
                                                                            reduction = 'none',
                                                                            layernorm_idx=1,
                                                                        )
                        loss_scene = answer_output_scene.loss
                        loss_scene = loss_scene.sum() / image_embeds.size(0)
                        loss = loss + loss_scene # TODO: add balance factor?

                    question_output.last_hidden_state = self.fuse_2d3d(question_output, question_output_scene)
            
                return loss, question_output.last_hidden_state, question_attention_mask # or return answer_output.last_hidden_state?
            
            else:
                last_hidden_state = question_output.last_hidden_state
                logits = self.answer_cls(last_hidden_state[:,0,:]) 
                answer_score_2d = logits.clone()

                if self.scene_feature_position in ["parallel++", "parallelshare", "paralleltwin"]:
                    # predict from question_output_scene
                    answer_score_scene = self.answer_cls(question_output_scene.last_hidden_state[:,0,:])

                    question_output.last_hidden_state = self.fuse_2d3d(question_output, question_output_scene)

                    if self.use_scene_classifier_2d3d:
                        answer_score_2d3d = self.answer_cls_2d3d(question_output.last_hidden_state[:,0,:])
                        logits = (logits + answer_score_scene + answer_score_2d3d) / 3
                    else:
                        answer_score_2d3d = None
                        logits = (logits + answer_score_scene) / 2
            
                    return (logits, answer_score_2d, answer_score_scene, answer_score_2d3d), question_output.last_hidden_state, question_attention_mask # or return answer_output.last_hidden_state?
            
            
            return logits, last_hidden_state, question_attention_mask # [batch_size, num_answers]
        else:
            if not self.use_text_decoder:
                last_hidden_state = question_output.last_hidden_state
                logits = self.answer_cls(last_hidden_state[:,0,:]) 
                answer_score_2d = logits.clone()

                if self.scene_feature_position in ["parallel++", "parallelshare", "paralleltwin"]:
                    # predict from question_output_scene
                    answer_score_scene = self.answer_cls(question_output_scene.last_hidden_state[:,0,:])

                    question_output.last_hidden_state = self.fuse_2d3d(question_output, question_output_scene)

                    if self.use_scene_classifier_2d3d:
                        answer_score_2d3d = self.answer_cls_2d3d(question_output.last_hidden_state[:,0,:])
                        logits = (logits + answer_score_scene + answer_score_2d3d) / 3
                    else:
                        answer_score_2d3d = None
                        logits = (logits + answer_score_scene) / 2
            
                    return (logits, answer_score_2d, answer_score_scene, answer_score_2d3d), question_output.last_hidden_state, question_attention_mask # or return answer_output.last_hidden_state?
                
                return logits, last_hidden_state, question_attention_mask
            
            if inference=='generate':
                num_beams = 5
                question_states = concat_repeat(question_output.last_hidden_state, question_output_scene.last_hidden_state, num_beams)

                # question_atts = concat_repeat(question_attention_mask, question_attention_mask, num_beams)
                question_atts = torch.repeat_interleave(question_attention_mask, 2 * num_beams, dim=0)
                model_kwargs = {"encoder_hidden_states": question_states, "encoder_attention_mask":question_atts}
                
                bos_ids = torch.full((image_embeds.size(0),1),fill_value=self.tokenizer.bos_token_id,device=image_embeds.device)
                
                outputs = self.text_decoder.generate(input_ids=bos_ids,
                                                     max_length=20,
                                                     min_length=1,
                                                     num_beams=num_beams * 2, # one group of beam for 3D, one for 2D
                                                     eos_token_id=self.tokenizer.sep_token_id,
                                                     pad_token_id=self.tokenizer.pad_token_id, 
                                                     **model_kwargs)
                
                answers = []    
                for output in outputs:
                    answer = self.tokenizer.decode(output, skip_special_tokens=True)    
                    answers.append(answer)
                return answers, self.fuse_2d3d(question_output, question_output_scene), question_attention_mask

            # rank answers - equiv to one-step beam search + brute force decodingg
            # NOTE: here answer should be the all_answer list
            assert answer is not None, "answer must be specified if use text decoder (free-form answer mode)"
            answer = self.tokenizer(answer, padding='longest', return_tensors="pt").to(image.device) 
            answer.input_ids[:,0] = self.tokenizer.bos_token_id
            ans_idx, ans_score = self.rank_answer(question_output.last_hidden_state, question_attention_mask, 
                                        answer.input_ids, answer.attention_mask, k_test) 
            sorted_idx = ans_score.argsort(dim=-1, descending=True)

            all_answer_score = torch.zeros(
                [image_embeds.size(0), answer.input_ids.size(0)], device=image_embeds.device
            ) # [batch_size, num_answers]

            # ic(all_answer_score.shape, ans_idx.shape, ans_score.shape)
            for i in range(ans_score.size(0)):
                all_answer_score[i % image_embeds.size(0)][ans_idx[i]] += ans_score[i] # loglikelihood to likelihood/prob?


            # predict from question_output_scene, and simple ensemble
            if self.scene_feature_position in ["parallel++", "parallelshare", "paralleltwin"]:
                all_answer_score_2d = all_answer_score.clone()
                answer_score_2d3d = None
                if self.use_scene_classifier:
                    all_answer_score_scene = torch.softmax(self.answer_cls(question_output_scene.last_hidden_state[:,0,:]), dim=-1)
                    # fill 0 with -1e4
                    all_answer_score = torch.where(
                        all_answer_score == 0, -1e4, all_answer_score
                    )
                    # pad all_answer_score to the same size as all_answer_score_scene
                    if all_answer_score.size(1) < all_answer_score_scene.size(1):
                        all_answer_score = F.pad(all_answer_score, (0, all_answer_score_scene.size(1) - all_answer_score.size(1)), "constant", -1e4)

                    all_answer_score = torch.softmax(all_answer_score, dim=-1)
                    
                    # if data_dict.get("iteration", 0) % 100 == 0:
                    if (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or not torch.distributed.is_initialized():
                        # 1% of the time, print the all_answer_score
                        if random.random() < 0.01:
                            print("-"*20)
                            print("all_answer_score_2d:", all_answer_score[:,:200].detach().cpu().tolist())
                            print("all_answer_score_scene:", all_answer_score_scene[:,:200].detach().cpu().tolist())
                            # print("all_answer_score:", all_answer_score.detach().cpu().tolist())
                            print("-"*20)

                    if self.use_scene_classifier_2d3d:
                        answer_score_2d3d = torch.softmax(self.answer_cls_2d3d(question_output.last_hidden_state[:,0,:]), dim=-1)
                        all_answer_score = (all_answer_score + all_answer_score_scene + answer_score_2d3d) / 3
                    else:
                        all_answer_score = (all_answer_score + all_answer_score_scene) / 2

                else:
                    all_answer_score_scene = torch.zeros(
                        [image_embeds.size(0), answer.input_ids.size(0)], device=image_embeds.device
                    )

                    ans_idx_scene, ans_score_scene = self.rank_answer(question_output_scene.last_hidden_state, question_attention_mask,
                                                            answer.input_ids, answer.attention_mask, k_test, use_scene=True)

                    for i in range(ans_score_scene.size(0)):
                        all_answer_score_scene[i % image_embeds.size(0)][ans_idx_scene[i]] += ans_score_scene[i]
                    
                    # prediction ensemble
                    all_answer_score = torch.where(
                        all_answer_score == 0, -1e4, all_answer_score
                    )
                    all_answer_score_scene = torch.where(
                        all_answer_score_scene == 0, -1e4, all_answer_score_scene
                    )
                    all_answer_score = all_answer_score.exp() + (all_answer_score_scene * 1.05).exp() # add exp-loglikelihood == add likelihood/prob
                    

                all_answer_score_2d = torch.where(
                    all_answer_score_2d == 0, -1e4, all_answer_score_2d
                )

                # simple fuse 2d3d for ref obj prediction
                question_output.last_hidden_state = self.fuse_2d3d(question_output, question_output_scene)
                return question_output.last_hidden_state, (all_answer_score, all_answer_score_scene, all_answer_score_2d, answer_score_2d3d), question_attention_mask
            
            all_answer_score = torch.where(
                all_answer_score == 0, -1e4, all_answer_score
            )
            return question_output.last_hidden_state, all_answer_score, question_attention_mask
    
    def fuse_2d3d(self, question_output, question_output_scene):
        question_feat_2d = self.lowrank_2d(question_output.last_hidden_state)
        question_feat_3d = self.lowrank_3d(question_output_scene.last_hidden_state)
        output = self.bilinear_fusion(question_feat_2d, question_feat_3d) + \
                        (question_output.last_hidden_state + question_output_scene.last_hidden_state) / 2.0
        return output
            
    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k, use_scene=False):
        if use_scene and self.scene_feature_position in ["parallel++", "paralleltwin"]:
            text_decoder = self.text_decoder_scene
        else:
            text_decoder = self.text_decoder

        num_ques = question_states.size(0)
        start_ids = answer_ids[0,0].repeat(num_ques,1) # bos token
        
        start_output = text_decoder(start_ids, 
                                         encoder_hidden_states = question_states,
                                         encoder_attention_mask = question_atts,                                      
                                         return_dict = True,
                                         reduction = 'none',
                                         layernorm_idx=1 if use_scene and self.scene_feature_position == "parallelshare" else 0,
                                         )              
        logits = start_output.logits[:,0,:] # first token's logit
        
        # topk_probs: top-k probability 
        # topk_ids: [num_question, k]        
        answer_first_token = answer_ids[:,1]
        prob_first_token = F.softmax(logits,dim=1).index_select(dim=1, index=answer_first_token) 
        k = min(prob_first_token.size(1), k) # in case k > num_answers
        topk_probs, topk_ids = prob_first_token.topk(k,dim=1) 
        
        # answer input: [num_question*k, answer_len]                 
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids,dim=0)  
        input_atts = torch.cat(input_atts,dim=0)  

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)
        
        output = text_decoder(input_ids, 
                                   attention_mask = input_atts, 
                                   encoder_hidden_states = question_states,
                                   encoder_attention_mask = question_atts,     
                                   labels = targets_ids,
                                   return_dict = True, 
                                   reduction = 'none',
                                   layernorm_idx=1 if use_scene and self.scene_feature_position == "parallelshare" else 0
                                   )   
        
        log_probs_sum = -output.loss
        log_probs_sum = log_probs_sum.view(num_ques,k)

        max_topk_ids = log_probs_sum.argmax(dim=1) 
        max_ids = topk_ids[max_topk_ids>=0,max_topk_ids]

        # return max_ids
        return topk_ids, log_probs_sum
    
    
def blip_vqa3d(pretrained='', random_init_blip=False, **kwargs) -> "BLIP_VQA3D":
    model = BLIP_VQA3D(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
#         assert(len(msg.missing_keys)==0)
        model.copy_weights()
        model.pretrained = pretrained
        print(msg)
    if random_init_blip:
        print("resetting weights in BLIP encoder-decoder modules")
        # random re-init encoder-decoder modules
        model.text_encoder.apply(weight_reset)
        model.text_decoder.apply(weight_reset)
        try:
            model.text_encoder_scene.apply(weight_reset)
            model.text_decoder_scene.apply(weight_reset)
        except AttributeError:
            pass
    model.save_state_dict()
    return model  


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))    
        
        