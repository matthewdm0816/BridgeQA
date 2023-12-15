import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mcan_module import MCAN_ED, AttFlat, LayerNorm, MCAN_E, AttFlat, LayerNorm, SA, SGA, MLP
from models.backbone_module import Pointnet2Backbone
from models.voting_module import VotingModule
from models.proposal_module import ProposalModule
from models.lang_module import LangModule, CLIPLangModule
# from models.vilt_vqa_3d import ViltVQA3D
# from utils.vlm_align_util import (
#     scene_bbox_to_2d_feat,
#     repeat_elements,
#     reverse_augment,
#     reverse_align,
#     jitter_bbox,
#     calculate_cube_corners,
#     calculate_overlap,
#     batch_iou,
#     corners_to_edges,
#     DSET_VIEWS_PATH,
#     reverse_align_simple,
# )
from lib.dataset import DC
# from models.seqmae import MaskedAutoencoderTransformer
import random
from time import time
from copy import deepcopy
from typing import *
# from transformers 
from icecream import ic
from utils.blip_utils import *
from utils.reset_weight import weight_reset
from transformers import AutoConfig
import torch.distributed as dist

def multilabel_onehot(multiple_labels, bs, num_labels):
    ones = torch.ones(bs, num_labels).to(multiple_labels).float()
    onehot = torch.zeros(bs, num_labels).to(ones).float()
    onehot.scatter_(dim=1, index=multiple_labels, src=ones)
    return onehot.float()



class ScanQA(nn.Module):
    def __init__(
        self,
        num_answers,
        # proposal
        num_object_class,
        input_feature_dim,
        num_heading_bin,
        num_size_cluster,
        mean_size_arr,
        num_proposal=256,
        vote_factor=1,
        sampling="vote_fps",
        seed_feat_dim=256,
        proposal_size=128,
        pointnet_width=1,
        pointnet_depth=2,
        vote_radius=0.3,
        vote_nsample=16,
        # qa
        # answer_cls_loss="ce",
        answer_pdrop=0.3,
        mcan_num_layers=2,
        mcan_num_heads=8,
        mcan_pdrop=0.1,
        mcan_flat_mlp_size=512,
        mcan_flat_glimpses=1,
        mcan_flat_out_size=1024,
        # lang
        lang_use_bidir=False,
        lang_num_layers=1,
        lang_emb_size=300,
        lang_pdrop=0.1,
        bert_model_name=None,
        freeze_bert=False,
        finetune_bert_last_layer=False,
        # common
        hidden_size=128,
        # option
        use_object_mask=False,
        use_lang_cls=False,
        use_reference=False,
        use_answer=False,
        use_extra_obj_encoder=False,
        jitter_bbox=False,
        att_pdrop=0.3,  # temporarily, same with original paper
        att_drop_topk=100,
        save_pred=False,
        visualize_bbox=False,
        image_size=512,
        use_clip_lang=False,
        clip_model_name="ViT-L-14",
        clip_ckpt_name="laion2b_s32b_b82k",
        use_blip=False,
        votenet_ckpt="",
        use_text_decoder=True,
        all_answers=None,
        stage="",
        dset_views_path="",
        scene_feature_position="",
        use_scene_weight=False,
        i2tfile="",
        i2tfile_eval="",
        no_scene=False,
        scene_feature_type="full",
        first_stage_ckpt_path="",
        depth_fusion=False,
        use_scene_classifier=False,
        med_config='/home/mowentao/scratch/BLIP/configs/med_config.json',
        use_scene_classifier_2d3d=False,
        not_copy_weights=False,
        scene_encoder_layers=2,
        mix_tokens=False,
        share_decoder=False,
        num_hidden_layers_twin=None,
        encoder_layers=None,
        decoder_layers=None,
        random_init_blip=False,
        use_vilt=False,
    ):
        super().__init__()


        # Option
        self.use_object_mask = use_object_mask
        self.use_lang_cls = use_lang_cls
        self.use_reference = use_reference
        self.use_answer = use_answer

        self.stage = stage
        
        self.visualize_bbox = visualize_bbox
        self.image_size = image_size
        
        self.jitter_bbox = jitter_bbox
        self.att_pdrop = att_pdrop
        self.att_drop_topk = att_drop_topk

        self.save_pred = save_pred


        self.no_scene = no_scene
        self.scene_feature_type = scene_feature_type

        self.first_stage_ckpt_path = first_stage_ckpt_path

        lang_size = hidden_size * (1 + lang_use_bidir)

        # --- Load BLIP model ---
        self.use_blip = use_blip
        self.use_vilt = use_vilt
        if use_blip and self.stage != "DET":
            if self.scene_feature_type == "full":
                scene_feature_size = hidden_size
            else:
                raise NotImplementedError
            # use GT scene feature, for debug only
            # elif self.scene_feature_type == "locsem":
            #     scene_feature_size = 6 + 1 # location + semantic label
            # elif self.scene_feature_type == "locsemfeat":
            #     scene_feature_size = 6 + 1 + hidden_size # location + semantic label + feature

            self.blip_model, _ = get_blip_model_simple(
                alternative_ckpt="",
                # num_answers=3000 if use_text_decoder else num_answers, 
                num_answers=num_answers,
                scene_size=scene_feature_size, 
                answer_pdrop=answer_pdrop,
                use_text_decoder=use_text_decoder,
                scene_feature_position=scene_feature_position,
                use_scene_weight=use_scene_weight,
                use_scene_classifier=use_scene_classifier,
                med_config=med_config,
                use_scene_classifier_2d3d=use_scene_classifier_2d3d,
                not_copy_weights=not_copy_weights,
                scene_encoder_layers=scene_encoder_layers,
                mix_tokens=mix_tokens,
                share_decoder=share_decoder,
                num_hidden_layers_twin=num_hidden_layers_twin,
                encoder_layers=encoder_layers,
                decoder_layers=decoder_layers,
                random_init_blip=random_init_blip,
            )
            self.blip_model.train()

        # --- Detector
        # Object detection

        self.detection_backbone = Pointnet2Backbone(
            input_feature_dim=input_feature_dim,
            width=pointnet_width,
            depth=pointnet_depth,
            seed_feat_dim=seed_feat_dim,
        )
        # Hough voting
        self.voting_net = VotingModule(vote_factor, seed_feat_dim)

        # Vote aggregation and object proposal
        self.proposal_net = ProposalModule(
            num_object_class,
            num_heading_bin,
            num_size_cluster,
            mean_size_arr,
            num_proposal,
            sampling,
            seed_feat_dim=seed_feat_dim,
            proposal_size=proposal_size,
            radius=vote_radius,
            nsample=vote_nsample,
        )

        
        self.object_feat_linear = nn.Sequential(
            nn.Linear(proposal_size, hidden_size), nn.GELU()
        )

        if use_blip:
            self.enc_list_o = nn.ModuleList([SA(hidden_size, mcan_num_heads, mcan_pdrop) for _ in range(mcan_num_layers)])

            if use_text_decoder:
                assert all_answers is not None
                self.all_answers = all_answers

            if use_vilt:
                blip_enc_size = self.blip_model.config.hidden_size
            else:
                blip_enc_size = self.blip_model.text_encoder.config.hidden_size
            self.lang_cls = nn.Sequential(
                nn.Linear(blip_enc_size, hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, num_object_class),
            )

            # Esitimate confidence
            self.object_cls = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, 1),
            )
            self.linear_blip_to_object = nn.Linear(blip_enc_size, hidden_size)
            self.dec_list_qo = nn.ModuleList([SGA(hidden_size, mcan_num_heads, mcan_pdrop) for _ in range(mcan_num_layers)])


        else:
            # Esitimate confidence
            self.object_cls = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, 1),
            )
            # Language encoding
            # if use_clip_lang:
            #     self.lang_net = CLIPLangModule(
            #         clip_model_name=clip_model_name,
            #         clip_ckpt_name=clip_ckpt_name,
            #         output_size=lang_size,
            #     )
            # else:
            if _not_use_clip_lang := True:
                self.lang_net = LangModule(
                    num_object_class,
                    use_lang_classifier=False,
                    use_bidir=lang_use_bidir,
                    num_layers=lang_num_layers,
                    emb_size=lang_emb_size,
                    hidden_size=hidden_size,
                    pdrop=lang_pdrop,
                    bert_model_name=bert_model_name,
                    freeze_bert=freeze_bert,
                    finetune_bert_last_layer=finetune_bert_last_layer,
                )
            
            # Feature projection
            self.lang_feat_linear = nn.Sequential(
                nn.Linear(lang_size, hidden_size), nn.GELU()
            )

            # Language classifier [actually object affordance category classifier]
            self.lang_cls = nn.Sequential(
                nn.Linear(mcan_flat_out_size, hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, num_object_class),
            )

            # QA head
            self.attflat_visual = AttFlat(
                hidden_size, mcan_flat_mlp_size, mcan_flat_glimpses, mcan_flat_out_size, 0.1
            )
            self.attflat_lang = AttFlat(
                hidden_size, mcan_flat_mlp_size, mcan_flat_glimpses, mcan_flat_out_size, 0.1
            )
            self.answer_cls = nn.Sequential(
                nn.Linear(mcan_flat_out_size, hidden_size),
                nn.GELU(),
                nn.Dropout(answer_pdrop),
                nn.Linear(hidden_size, num_answers),
            )

        

        # Extra Object Encoder
        self.use_extra_obj_encoder = use_extra_obj_encoder
        if use_extra_obj_encoder:
            self.object_xencoder = MCAN_E(
                hidden_size,
                num_heads=mcan_num_heads,
                num_layers=mcan_num_layers,
                pdrop=mcan_pdrop,
            )
        else:
            self.object_xencoder = None

        # Fusion backbone
        self.fusion_backbone = MCAN_ED(
            hidden_size,
            num_heads=mcan_num_heads,
            num_layers=mcan_num_layers,
            pdrop=mcan_pdrop,
        )
        self.fusion_norm = LayerNorm(mcan_flat_out_size)

        # --- load pretrained votenet
        if votenet_ckpt != "":
            print("loading pretrained votenet from {}".format(votenet_ckpt))
            votenet_dict = torch.load(votenet_ckpt)
            # votenet_dict = {k.replace("module.", ""): v for k, v in votenet_dict.items()}
            self.load_state_dict(votenet_dict, strict=False)


    def compute_object_assignment(self, data_dict):
        """Compute objectness loss for the proposals.

        Args:
            data_dict: dict (read-only)

        Returns:
            objectness_loss: scalar Tensor
            objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
            objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
            object_assignment: (batch_size, num_seed) Tensor with long int
                within [0,num_gt_object-1]
        """
        from lib.loss_helper import NEAR_THRESHOLD, FAR_THRESHOLD
        from utils.nn_distance import nn_distance

        # Associate proposal and GT objects by point-to-point distances
        aggregated_vote_xyz = data_dict["aggregated_vote_xyz"]
        gt_center = data_dict["center_label"][:, :, 0:3]
        B = gt_center.shape[0]
        K = aggregated_vote_xyz.shape[1]
        K2 = gt_center.shape[1]
        dist1, ind1, dist2, _ = nn_distance(
            aggregated_vote_xyz, gt_center
        )  # dist1: BxK, dist2: BxK2

        # Generate objectness label and mask
        # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
        # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
        euclidean_dist1 = torch.sqrt(dist1 + 1e-6)
        objectness_label = torch.zeros((B, K), dtype=torch.long).cuda()
        objectness_label[euclidean_dist1 < NEAR_THRESHOLD] = 1

        object_assignment = ind1

        return objectness_label, object_assignment
    

    def load_image(self, data_dict, device):
        images = data_dict["images"].to(device)
        poses = data_dict["poses"].to(device) # batch, num_view, 4, 4
        depths = data_dict["depths"].to(device)
        poses = poses.flatten(-2, -1) # batch, num_view, 16

        return images, poses, depths
    
    def load_pretrained(self):
        if self.stage == "VQA" and self.first_stage_ckpt_path != "":
            model_dict = self.state_dict()
            pretrained_dict = torch.load(self.first_stage_ckpt_path, map_location="cpu")
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(pretrained_dict, "module.")

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            print(list(pretrained_dict.keys()))
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.load_state_dict(model_dict)
    
    def reinit_params(self):
        if self.use_blip:
            self.blip_model.save_state_dict() # blip save
        self.apply(weight_reset) # reinit all
        self.load_pretrained() # load detector
        if self.use_blip:
            self.blip_model.reinit_params() # blip load
        


    def forward(self, data_dict):
        
        image_feats = None
        device = data_dict["point_clouds"].device
        ### 2D Visual Encoder ###
        if self.use_blip:
            scene_ids = data_dict["scene_id_str"]
            question_ids = data_dict["question_id_str"]
            images, poses, depths = self.load_image(data_dict, device)
            
            

        #######################################
        #                                     #
        #           DETECTION BRANCH          #
        #                                     #
        #######################################

        # --------- HOUGH VOTING ---------
        data_dict = self.detection_backbone(data_dict)

        # --------- HOUGH VOTING ---------
        xyz = data_dict["fp2_xyz"]
        features = data_dict[
            "fp2_features"
        ]  # batch_size, seed_feature_dim, num_seed, (16, 256, 1024)
        data_dict["seed_inds"] = data_dict["fp2_inds"]
        data_dict["seed_xyz"] = xyz

        data_dict["seed_features"] = features
        xyz, features = self.voting_net(
            xyz, features
        )  # batch_size, vote_feature_dim, num_seed * vote_factor, (16, 256, 1024)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        data_dict["vote_xyz"] = xyz
        data_dict["vote_features"] = features

        # --------- PROPOSAL GENERATION ---------
        data_dict["jitter_bbox"] = self.jitter_bbox
        data_dict = self.proposal_net(xyz, features, data_dict)

        # unpack outputs from detection branch
        object_feat = data_dict[
            "aggregated_vote_features"
        ]  # batch_size, num_proposal, proposal_size (128)

        if self.use_object_mask:
            object_mask = (
                ~data_dict["bbox_mask"].bool().detach()
            )  #  # batch, num_proposals
        else:
            object_mask = None
        
        if object_mask.dim() == 2:
            object_mask = object_mask.unsqueeze(1).unsqueeze(2) # [B, N_proposal] => [B, 1, 1, N_proposal]

        
        object_feat = self.object_feat_linear(
            object_feat
        )  # batch_size, num_proposal, hidden_size

        if _not_use_mae := True:
            data_dict["mae_loss"] = torch.zeros(1).to(object_feat.device)

        if self.use_extra_obj_encoder:
            object_feat = self.object_xencoder(object_feat, object_mask)

        ########################################
        #                                      #
        #             VLM Alignment            #
        #                                      #
        ########################################
        # --- QA
        if not self.use_blip:
            # This mainly used for detector pretrain, we don't need to compute QA branch, so don't need to load BLIP
            #######################################
            #                                     #
            #           LANGUAGE BRANCH           #
            #                                     #
            #######################################
            # --------- LANGUAGE ENCODING ---------
            data_dict = self.lang_net(data_dict)
            # --------- QA BACKBONE ---------
            #######################################
            #                                     #
            #             QA BACKBONE             #
            #                                     #
            #######################################

            # unpack outputs from question encoding branch
            lang_feat = data_dict["lang_out"]  
            # word embeddings after LSTM (batch_size, num_words(max_question_length), hidden_size * num_dir
            lang_mask = data_dict["lang_mask"]  
            # word attetion (batch, num_words)
            if lang_mask.dim() == 2:
                lang_mask = lang_mask.unsqueeze(1).unsqueeze(2)

            # Pre-process Lanauge & Image Feature
            lang_feat = self.lang_feat_linear(
                lang_feat
            )  # batch_size, num_words, hidden_size
            # QA Backbone (Fusion network)
            lang_feat, object_feat = self.fusion_backbone(
                lang_feat,
                object_feat,
                lang_mask,
                object_mask,
                # self.att_pdrop if data_dict["phase"] == "train" else 0,
                # self.att_drop_topk,
            )
            # object_feat: batch_size, num_proposal, hidden_size
            # lang_feat: batch_size, num_words, hidden_size

            #######################################
            #                                     #
            #          PROPOSAL MATCHING          #
            #                                     #
            #######################################
            if self.use_reference:
                #  data_dict["cluster_ref"]:
                #  tensor([[-0.2910, -0.2910, -0.1096],
                #          [0.7795, -0.2910,  1.2384]]
                # mask out invalid proposals
                object_conf_feat = object_feat * data_dict["objectness_scores"].max(2)[
                    1
                ].float().unsqueeze(2)
                data_dict["cluster_ref"] = self.object_cls(object_conf_feat).squeeze(-1) # [B, num_proposal]

            lang_feat = self.attflat_lang(lang_feat, lang_mask)

            # if data_dict["phase"] == "train" and self.att_pdrop > 0:
            #     object_mask &= (torch.rand(object_mask.shape, device=object_mask.device) > self.att_pdrop)

            object_feat = self.attflat_visual(
                object_feat,
                object_mask,
                self.att_pdrop if data_dict["phase"] == "train" else 0,
                self.att_drop_topk,
            )

            fuse_feat = self.fusion_norm(
                lang_feat + object_feat
            )  # batch, mcan_flat_out_size

            data_dict["fuse_feat"] = fuse_feat.clone()

            # if self.fuse_vlm:
            #     fuse_feat = self.fusion_norm(fuse_feat + blip_fused_feat.mean(dim=1))

            #######################################
            #                                     #
            #           LANGUAGE BRANCH           #
            #                                     #
            #######################################
            if self.use_lang_cls:
                data_dict["lang_scores"] = self.lang_cls(
                    fuse_feat
                )  # batch_size, num_object_classe

            #######################################
            #                                     #
            #          QUESTION ANSERING          #
            #                                     #
            #######################################
            if self.use_answer:
                data_dict["answer_scores"] = self.answer_cls(
                    fuse_feat
                )  # batch_size, num_answers
                # if data_dict["phase"] != "train":
                #     # output answer
                #     data_dict["pred_answers"] = data_dict["answer_scores"].argmax(dim=-1)

            if self.use_soft_label_align:
                if not (self.soft_label_on_train and data_dict["phase"] != "train"):
                    question_ids = data_dict["question_id_str"]
                    data_dict["soft_label"] = torch.stack([self.soft_label[qid] for qid in question_ids], dim=0).to(fuse_feat) # [B, N_ans]
                    # print(data_dict["answer_scores"][0], data_dict["soft_label"][0])
        
        else:
            # ---VQA branch
            if self.stage != "DET":
                questions = data_dict["question"]
                question_ids = data_dict["question_id_str"]
                if isinstance(questions, torch.Tensor):
                    questions = questions.detach().cpu().tolist()
                scene_names = data_dict["scene_id_str"]
                answers = data_dict["answers"]
                answer = [
                    random.choice(concat_answer.split("###")) for concat_answer in answers
                ]
                
                # if image_feats is None:
                #     images, poses, depths = self.load_image(data_dict, scene_names, question_ids, device)

                # O-former
                if self.scene_feature_type == "full":
                    object_feat_for_2d = object_feat.clone()
                    # for enc in self.enc_list_o:
                    #     object_feat_for_2d = enc(object_feat_for_2d, object_mask) # batch, num_proposal, hidden_size
                    object_mask_for_2d = ~object_mask # for BLIP/Transformers, 1 for valid, 0 for invalid
                    object_mask_for_2d = object_mask_for_2d.squeeze(1).squeeze(1) # [B, 1, 1, N_proposal] -> [B, N_proposal]
                # elif self.scene_feature_type == "locsem":
                #     (
                #         objectness_label,
                #         object_assignment,
                #     ) = self.compute_object_assignment(data_dict)
                #     sem_cls_label = torch.gather(data_dict['sem_cls_label'], 1, object_assignment)
                #     target_bboxes = torch.gather(data_dict['target_bboxes'], 1, object_assignment.unsqueeze(-1).repeat(1,1,6)) # [B, N_bbox, 6] => [B, N_proposal, 6]
                #     box_label_mask = torch.gather(data_dict['box_label_mask'], 1, object_assignment) # [B, N_bbox] => [B, N_proposal], 1 for valid, 0 for invalid
                #     object_mask_for_2d = box_label_mask.to(object_mask) # [B, N_proposal]
                #     object_feat_for_2d = torch.cat([target_bboxes, sem_cls_label.unsqueeze(-1)], dim=-1) # [B, N_proposal, 6+1]
                # elif self.scene_feature_type == "locsemfeat":
                #     (
                #         objectness_label,
                #         object_assignment,
                #     ) = self.compute_object_assignment(data_dict)
                #     sem_cls_label = torch.gather(data_dict['sem_cls_label'], 1, object_assignment)
                #     target_bboxes = torch.gather(data_dict['target_bboxes'], 1, object_assignment.unsqueeze(-1).repeat(1,1,6)) # [B, N_bbox, 6] => [B, N_proposal, 6]
                #     box_label_mask = torch.gather(data_dict['box_label_mask'], 1, object_assignment) # [B, N_bbox] => [B, N_proposal], 1 for valid, 0 for invalid
                #     object_mask_for_2d = box_label_mask.to(object_mask) # [B, N_proposal]
                #     object_mask_for_2d = object_mask_for_2d & (~object_mask.squeeze(1).squeeze(1)) # for BLIP/Transformers, 1 for valid, 0 for invalid
                #     object_feat_for_2d = torch.cat([target_bboxes, sem_cls_label.unsqueeze(-1)], dim=-1) # [B, N_proposal, 6+1]
                #     object_feat_for_2d = torch.cat([object_feat_for_2d, object_feat], dim=-1) # [B, N_proposal, 6+1+hidden_size]
                else:
                    raise NotImplementedError

                depth_map = None

                if self.use_vilt:
                    print(data_dict["images_raw"].shape)
                    print("use vilt")
                    images_for_2d = data_dict["images_raw"][:, 0]
                else:
                    images_for_2d = images[:, 0] # 0 ~ take the first view

                inference = "generate" if data_dict.get("open_ended", False) else "rank"
                output = self.blip_model(
                    # images[:, 0], # 0 ~ take the first view
                    images_for_2d,
                    questions, 
                    scene_object_embeds=object_feat_for_2d, 
                    scene_object_mask=object_mask_for_2d,
                    answer=answer 
                        if (data_dict["phase"] == "train" or not self.blip_model.use_text_decoder) 
                        else self.all_answers[:4500]
                        ,
                    train=data_dict["phase"] == "train",
                    k_test=256,
                    image_pose=poses[:, 0],
                    image_embeds=image_feats,
                    depth_map=depth_map,
                    inference=inference,
                    data_dict=data_dict,
                )
                if inference == "generate":
                    data_dict["open_ended_answer"] = output[0]
                    print(data_dict["open_ended_answer"])
                    return data_dict
                if self.blip_model.use_text_decoder:
                    if data_dict["phase"] == "train":
                        loss, fused_feat, fused_mask = output
                        if isinstance(loss, tuple):
                            loss, answer_scores_scene, answer_scores_2d3d = loss
                            if len(self.all_answers) > answer_scores_scene.shape[1]:
                                answer_scores_scene = F.pad(answer_scores_scene, (0, len(self.all_answers) - answer_scores_scene.shape[1]), value=-1e4)
                            data_dict["answer_scores_scene"] = answer_scores_scene # for scene loss calculation
                            if answer_scores_2d3d is not None:
                                if len(self.all_answers) > answer_scores_2d3d.shape[1]:
                                    answer_scores_scene = F.pad(answer_scores_2d3d, (0, len(self.all_answers) - answer_scores_2d3d.shape[1]), value=-1e4)
                                data_dict["answer_scores_2d3d"] = answer_scores_2d3d # for 2d3d loss calculation
                        else:
                            answer_scores_scene = None
                        data_dict["decoder_loss"] = loss # LM loss
                        data_dict["answer_scores"] = data_dict["answer_cat_scores"] # GT score
                    else:
                        fused_feat, answer_scores, fused_mask = output
                        # already log-likelihood
                        # pad answer scores to num_answer
                        if isinstance(answer_scores, tuple):
                            answer_scores, answer_scores_scene, answer_scores_2d, answer_scores_2d3d = answer_scores
                            # log scores
                            
                        else:
                            answer_scores_scene = None
                            answer_scores_2d = None
                            answer_scores_2d3d = None
                        if len(self.all_answers) > answer_scores.shape[1]:
                            answer_scores = F.pad(answer_scores, (0, len(self.all_answers) - answer_scores.shape[1]), value=-1e4)
                        data_dict["answer_scores"] = answer_scores

                        if answer_scores_scene is not None:
                            if len(self.all_answers) > answer_scores_scene.shape[1]:
                                answer_scores_scene = F.pad(answer_scores_scene, (0, len(self.all_answers) - answer_scores_scene.shape[1]), value=-1e4)
                            data_dict["answer_scores_scene"] = answer_scores_scene
                            if len(self.all_answers) > answer_scores_2d.shape[1]:
                                answer_scores_2d = F.pad(answer_scores_2d, (0, len(self.all_answers) - answer_scores_2d.shape[1]), value=-1e4)
                            data_dict["answer_scores_2d"] = answer_scores_2d

                        if answer_scores_2d3d is not None:
                            if len(self.all_answers) > answer_scores_2d3d.shape[1]:
                                answer_scores_2d3d = F.pad(answer_scores_2d3d, (0, len(self.all_answers) - answer_scores_2d3d.shape[1]), value=-1e4)
                            data_dict["answer_scores_2d3d"] = answer_scores_2d3d
                        
                else:
                    answer_scores, fused_feat, fused_mask = output
                    if isinstance(answer_scores, tuple):
                        answer_scores, answer_scores_scene, answer_scores_2d, answer_scores_2d3d = answer_scores
                    else:
                        answer_scores_scene = None
                        answer_scores_2d = None
                        answer_scores_2d3d = None
                    # answer_scores = F.pad(answer_scores, (0, len(self.all_answers) - answer_scores.shape[1]), value=-1e4)
                    data_dict["answer_scores"] = answer_scores
                    if answer_scores_scene is not None:
                        data_dict["answer_scores_scene"] = answer_scores_scene
                    if answer_scores_2d is not None:
                        data_dict["answer_scores_2d"] = answer_scores_2d
                    if answer_scores_2d3d is not None:
                        data_dict["answer_scores_2d3d"] = answer_scores_2d3d

                if self.use_lang_cls:
                    data_dict["lang_scores"] = self.lang_cls(fused_feat[:, 0, :])

                # if self.use_aux_situation:
                #     data_dict["aux_scores"] = self.aux_reg(fused_feat[:, 0, :])

                if self.use_reference:
                    #  data_dict["cluster_ref"]:
                    #  tensor([[-0.2910, -0.2910, -0.1096],
                    #          [0.7795, -0.2910,  1.2384]]
                    # mask out invalid proposals
                    fused_feat_for_crossatt = self.linear_blip_to_object(fused_feat)
                    # object_mask = object_mask.unsqueeze(1).unsqueeze(2).bool()
                    fused_mask = fused_mask.unsqueeze(1).unsqueeze(2).bool()
                    for dec in self.dec_list_qo:
                        object_feat = dec(object_feat, fused_feat_for_crossatt, ~object_mask, ~fused_mask, att_pdrop=None, att_drop_topk=None)
                    object_conf_feat = object_feat * data_dict["objectness_scores"].max(2)[
                        1
                    ].float().unsqueeze(2)
                    data_dict["cluster_ref"] = self.object_cls(object_conf_feat).squeeze(-1) # [B, num_proposal]

                

                # fused_feat ~ [B, L, D]

        return data_dict
