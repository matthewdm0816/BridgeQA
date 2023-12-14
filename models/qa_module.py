import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mcan_module import MCAN_ED, AttFlat, LayerNorm, MCAN_E, AttFlat, LayerNorm, SA, SGA, MLP
from models.backbone_module import Pointnet2Backbone
from models.voting_module import VotingModule
from models.proposal_module import ProposalModule
from models.lang_module import LangModule, CLIPLangModule
from models.vilt_vqa_3d import ViltVQA3D
from utils.vlm_align_util import (
    scene_bbox_to_2d_feat,
    repeat_elements,
    reverse_augment,
    reverse_align,
    jitter_bbox,
    calculate_cube_corners,
    calculate_overlap,
    batch_iou,
    corners_to_edges,
    DSET_VIEWS_PATH,
    reverse_align_simple,
)
from lib.dataset import DC
from models.seqmae import MaskedAutoencoderTransformer
import random
from time import time
from copy import deepcopy
from typing import *
# from transformers 
from icecream import ic
from utils.blip_utils import *
from utils.reset_weight import weight_reset
from transformers import AutoConfig

def multilabel_onehot(multiple_labels, bs, num_labels):
    ones = torch.ones(bs, num_labels).to(multiple_labels).float()
    onehot = torch.zeros(bs, num_labels).to(ones).float()
    onehot.scatter_(dim=1, index=multiple_labels, src=ones)
    return onehot.float()

class ResMLP(nn.Module):
    def __init__(
        self,
        fin: int,
        fmid: int,
        dropout=0.1,
    ):
        super().__init__()

        self.net = nn.ModuleList(
            [
                nn.Linear(fin, fmid),
                nn.LayerNorm(fmid),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fmid, fin),
                nn.LayerNorm(fin),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        )
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x) + x

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout=0.1):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.fc1 = ResMLP(input_dim, input_dim // 2, dropout=dropout)
        self.fc21 = nn.Linear(input_dim, latent_dim)
        self.fc22 = nn.Linear(input_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, input_dim)
        self.fc4 = ResMLP(input_dim, input_dim // 2, dropout=dropout)

    def encode(self, x):
        h1 = self.fc1(x)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = self.fc3(z)
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

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
        # vlm align
        use_vlm_align=False,
        replace_3d_feature=False,
        vlm_hidden_size=768,  # 1024 for vit-large
        image_feat_dict=None,
        grid_size=None,
        overlap_threshold=0.7,
        scene_view_map=None,
        objectness_threshold=0.3,
        align_topk=1,
        random_sample_topk=False,
        begin_align_epoch=0,
        align_fused_vlm=False,
        fuse_vlm=False,
        use_gt_obj_align=False,
        simple_align=False,
        use_extra_obj_encoder=False,
        use_contrastive=False,
        use_variational_aligner=False,
        vae_latent_size=128,
        contrastive_temperature=10.0,
        use_separate_vae=False,
        vae_dropout=0,
        use_vs=False,
        # hard-positive mining
        jitter_bbox=False,
        att_pdrop=0.3,  # temporarily, same with original paper
        att_drop_topk=100,
        # view selector
        use_selector=False,
        use_mae=False,
        save_pred=False,
        align_one_gt=False,
        use_soft_label_align=False,
        soft_label_on_train=True,
        soft_label_path=None,
        visualize_bbox=False,
        image_size=512,
        mae_twice=True,
        mask_ratio=0.75,
        recon_xyz=False,
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
        project_2d=False,
        first_stage_ckpt_path="",
        depth_fusion=False,
        # answer_vocab=None,
        use_scene_classifier=False,
        med_config='/home/mowentao/scratch/BLIP/configs/med_config.json',
        use_scene_classifier_2d3d=False,
        not_copy_weights=False,
        scene_encoder_layers=2,
        mix_tokens=False,
        share_decoder=False,
        num_hidden_layers_twin=None,
        grl=False,
        grl_weight=0.1,
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
        
        self.use_mae = use_mae
        self.mae_twice = mae_twice
        self.recon_xyz = recon_xyz

        self.use_vlm_align = use_vlm_align
        self.align_fused_vlm = align_fused_vlm
        self.fuse_vlm = fuse_vlm

        self.no_scene = no_scene
        self.scene_feature_type = scene_feature_type
        self.project_2d = project_2d

        self.first_stage_ckpt_path = first_stage_ckpt_path
        self.depth_fusion = depth_fusion

        lang_size = hidden_size * (1 + lang_use_bidir)

        # --- Load BLIP model ---
        self.use_blip = use_blip
        self.use_vilt = use_vilt
        if use_blip and self.stage != "DET":
            if self.scene_feature_type == "full":
                scene_feature_size = hidden_size
            elif self.scene_feature_type == "locsem":
                scene_feature_size = 6 + 1 # location + semantic label
            elif self.scene_feature_type == "locsemfeat":
                scene_feature_size = 6 + 1 + hidden_size # location + semantic label + feature

            # (self.images, self.images_eval), self.blip_model, (self.scene_view_map, self.scene_view_map_eval), (self.poses, self.poses_eval), (self.depths, self.depths_eval), _ \
            # = get_blip_model(
            #     i2tfile, 
            #     i2tfile_eval,
            #     topk=align_topk, 
            #     alternative_ckpt="", 
            #     dset_views_path=None if dset_views_path == "" else dset_views_path,
            #     num_answers=num_answers, 
            #     scene_size=scene_feature_size, 
            #     answer_pdrop=answer_pdrop,
            #     use_text_decoder=use_text_decoder,
            #     scene_feature_position=scene_feature_position,
            #     use_scene_weight=use_scene_weight,
            #     dset="scanqa",
            # )
            if use_vilt:
                vilt_config = AutoConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
                vilt_config.num_labels = num_answers
                vilt_config.scene_size = scene_feature_size
                vilt_config.scene_feature_position = scene_feature_position
                
                self.blip_model, loading_info = ViltVQA3D.from_pretrained("dandelin/vilt-b32-finetuned-vqa", config=vilt_config, output_loading_info=True, ignore_mismatched_sizes=True)
                print(loading_info)
                
                device = torch.device("cuda", dist.get_rank()) 
                self.blip_model.train()
                self.blip_model = self.blip_model.to(device)

            else:
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
                    grl=grl,
                    grl_weight=grl_weight,
                    encoder_layers=encoder_layers,
                    decoder_layers=decoder_layers,
                    random_init_blip=random_init_blip,
                )
                self.blip_model.train()

        # --- Detector
        # Ojbect detection
        if project_2d and use_blip:
            # input_feature_dim = input_feature_dim + self.blip_model.visual_encoder.num_features
            self.proj2d_linear = nn.Linear(self.blip_model.visual_encoder.num_features, input_feature_dim + 3)

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
            if use_clip_lang:
                self.lang_net = CLIPLangModule(
                    clip_model_name=clip_model_name,
                    clip_ckpt_name=clip_ckpt_name,
                    output_size=lang_size,
                )
            else:
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

        self.contrastive_temperature = contrastive_temperature
        self.use_contrastive = use_contrastive
        self.use_vs = use_vs

        if self.use_vs:
            # View Selector
            self.vs = MCAN_E(
                vlm_hidden_size,
                num_heads=mcan_num_heads,
                num_layers=mcan_num_layers,
                pdrop=mcan_pdrop,
            )
            self.attflat_vs = AttFlat(
                vlm_hidden_size, mcan_flat_mlp_size, mcan_flat_glimpses, vlm_hidden_size, 0.1
            )

        # VLM-alignment head
        self.overlap_threshold = overlap_threshold
        self.objectness_threshold = objectness_threshold
        self.grid_size = grid_size
        self.align_topk = align_topk
        self.random_sample_topk = random_sample_topk
        self.begin_align_epoch = begin_align_epoch
        self.use_gt_obj_align = use_gt_obj_align

        self.use_variational_aligner = use_variational_aligner
        self.use_separate_vae = use_separate_vae
        if use_vlm_align:
            
            self.scene_view_map: dict = scene_view_map
            # self.load_finetuned_blip = load_finetuned_blip
            # if self.load_finetuned_blip:
                
            # 22-12-25: simple 2 layer FC
            print(f"vit grid_size: {grid_size}")
            self.image_feat_dict = image_feat_dict
            if simple_align:
                self.object_aligner = nn.Sequential(
                    nn.Linear(hidden_size, vlm_hidden_size),
                )
            else:
                self.object_aligner = nn.Sequential(
                    nn.Linear(hidden_size, vlm_hidden_size),
                    nn.GELU(),
                    nn.Linear(vlm_hidden_size, vlm_hidden_size),
                )
            
            # self.vae_dropout = vae_dropout
            if self.use_variational_aligner:
                self.object_vae = VAE(vlm_hidden_size, vae_latent_size, dropout=vae_dropout)
                if self.use_separate_vae:
                    self.region_vae = VAE(vlm_hidden_size, vae_latent_size, dropout=vae_dropout)
                

            
            if align_fused_vlm or fuse_vlm:
                self.blip_fusion_backbone = MCAN_ED(
                    hidden_size,
                    num_heads=mcan_num_heads,
                    num_layers=2,
                    pdrop=mcan_pdrop,
                )
                self.blip_feat_linear = nn.Sequential(
                    nn.Linear(vlm_hidden_size, hidden_size), nn.GELU()
                )
                self.blip_attflat_visual = AttFlat(
                    hidden_size,
                    mcan_flat_mlp_size,
                    mcan_flat_glimpses,
                    mcan_flat_out_size,
                    0.1,
                )
                self.blip_attflat_lang = AttFlat(
                    hidden_size,
                    mcan_flat_mlp_size,
                    mcan_flat_glimpses,
                    mcan_flat_out_size,
                    0.1,
                )
                if use_selector:
                    self.selector_head = nn.Sequential(
                        nn.Linear(mcan_flat_out_size, hidden_size),
                        nn.GELU(),
                        nn.Dropout(answer_pdrop),
                        nn.Linear(hidden_size, 1),
                    )
                else:
                    self.selector_head = None

        if self.use_mae:
            default_config = MaskedAutoencoderTransformer.default_config()
            self.object_mae = MaskedAutoencoderTransformer(
                input_dim=hidden_size+3 if recon_xyz else hidden_size,
                input_max_len=num_proposal,
                recon_xyz=recon_xyz,
                **default_config,
            )
            self.object_mae_linear = nn.Linear(default_config["embed_dim"], hidden_size)
            self.mask_ratio = mask_ratio

        self.align_one_gt = align_one_gt

        # Soft label align with 2D VLM
        self.use_soft_label_align = use_soft_label_align
        self.soft_label_path = soft_label_path
        self.soft_label_on_train = soft_label_on_train
        from lib.dataset import Answer
        self.answer_vocab: Optional[Answer] = None
        self.replace_3d_feature = replace_3d_feature
        if self.replace_3d_feature:
            self.object_aligner_rev = nn.Linear(vlm_hidden_size, hidden_size)


        # --- load pretrained votenet
        if votenet_ckpt != "":
            print("loading pretrained votenet from {}".format(votenet_ckpt))
            votenet_dict = torch.load(votenet_ckpt)
            # votenet_dict = {k.replace("module.", ""): v for k, v in votenet_dict.items()}
            self.load_state_dict(votenet_dict, strict=False)


    def load_soft_label(self):
        if self.use_soft_label_align:
            print("loading soft label...")
            assert self.answer_vocab is not None
            import json
            tmp = json.load(open(self.soft_label_path, "r"))
            soft_label: dict[str, torch.Tensor] = {k: torch.tensor(v) for k, v in tmp["answer_score"].items()} # unsoftmaxed
            # if self.soft_label_on_train:
            #     soft_label = {k: v for k, v in soft_label.values() if k.startswith("train")}
            all_answer: list[str] = tmp["all_answer"]
            print(len(all_answer), len(self.answer_vocab.vocab), len(set(all_answer) - set(self.answer_vocab.vocab)))
            self.soft_label = {k: torch.zeros(len(self.answer_vocab)) for k, v in tmp["answer_score"].items()}
            new_idxs = torch.tensor([self.answer_vocab.stoi(answer) for answer in all_answer]).long()
            if (new_idxs < 0).any():
                print("found non-exist soft label, replacing to last category")
                new_idxs[new_idxs < 0] = len(self.answer_vocab) - 1 # for debug
            for k, v in soft_label.items():
                self.soft_label[k].scatter_(src=v, dim=-1, index=new_idxs)
                self.soft_label[k] = torch.where(self.soft_label[k].abs() < 1e-6, -1e2, self.soft_label[k]) # remaininig set to -inf
                # for idx, prob in enumerate(v):
                #     self.soft_label[k][new_idxs[idx]] = prob ~ v[idx]


    def forward_vs(self, region_view_feats: torch.Tensor, align_mask: torch.BoolTensor):
        ## region_view_feats : [B, K, N_proposal, F_vlm]    
        # align_mask : [B, K, N_proposal] 
        # returns [B, N_proposal, F_vlm] 
        print(region_view_feats.shape, align_mask.shape)
        assert len(region_view_feats.shape) == 4  
        B, _, N_prop, _ = region_view_feats.shape
        x = region_view_feats.transpose(1, 2).flatten(0, 1) # => [B * N_prop, K, F]
        x_mask = (~align_mask).transpose(1, 2).flatten(0, 1) # [B * N_prop, K], for mcan, 0 is "has token", 1 is masked off
        # x_mask = torch.ones(x.shape[:-1], device=region_view_feats.device) 
        x_mask = x_mask.unsqueeze(1).unsqueeze(2).bool()
        x = self.vs(x, x_mask)
        x = self.attflat_vs(x, x_mask)
        return x.view(B, N_prop, -1)

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
        # if self.use_contrastive:
        data_dict["use_contrastive"] = self.use_contrastive
        data_dict["contrastive_temperature"] = self.contrastive_temperature
        
        image_feats = None
        device = data_dict["point_clouds"].device
        ### 2D Visual Encoder ###
        if self.use_blip:
            scene_ids = data_dict["scene_id_str"]
            question_ids = data_dict["question_id_str"]
            images, poses, depths = self.load_image(data_dict, device)
            # images ~ [B, num_view, ...]
            print("reversing axis align")
            original_point_cloud = reverse_align_simple(
                points=data_dict["original_point_cloud"],
                align_mat=data_dict["axis_align_matrix"],
            )
            if self.project_2d:
                align_proj_feats = []
                align_proj_masks = []
                image_feats_grid_list = []
                projection_weight_list = []
                for idx in range(self.align_topk):
                    poses_idx = poses[:, idx].reshape(-1, 4, 4)
                    # if idx == 0 or not self.random_project:
                    image_feats_grid, projection_weight = self.blip_model(images[:, idx], None, embed_image=True)
                    # else:
                    #     # random take one of the topk
                    #     rand_
                    #     image_feats_grid, projection_weight = self.blip_model(images[:, idx], None, embed_image=True, random_project=True)
                    # [B, 1 + G * G, F], [B, 1]
                    if idx == 0:
                        image_feats = image_feats_grid.clone() # save best image feats
                    image_feats_grid = image_feats_grid.transpose(1, 2)[..., 1:] # [B, F, G * G], remove [CLS] token
                    grid_size = round(image_feats_grid.shape[-1] ** 0.5)
                    image_feats_grid = image_feats_grid.reshape(*image_feats_grid.shape[:2], grid_size, grid_size) # [B, F, G, G]
                    # proj_feats [B, num_points, F], proj_masks [B, num_points]
                    # image_feats_grid_list.append(image_feats_grid)
                    # projection_weight_list.append(projection_weight)

                # # softmax projection_weight over views
                # projection_weight = torch.stack(projection_weight_list, dim=1) # [B, num_views, 1]
                # projection_weight = torch.softmax(projection_weight, dim=1) # [B, num_views, 1]
                # projection_weight = projection_weight.unsqueeze(-1) # [B, num_views, 1, 1]
                
                # for idx in range(self.align_topk):
                    proj_feats, proj_masks = project_blip_to_pointcloud(
                        original_point_cloud, 
                        # image_feats_grid_list[idx] * projection_weight[:, idx],
                        image_feats_grid * projection_weight.unsqueeze(-1).unsqueeze(-1),
                        depths[:, idx], 
                        poses_idx, 
                        grad=True
                    )
                    align_proj_feats.append(proj_feats)
                    align_proj_masks.append(proj_masks.unsqueeze(-1))
                align_proj_feats = torch.stack(align_proj_feats, dim=1) # [B, num_views, num_points, F]
                align_proj_masks = torch.stack(align_proj_masks, dim=1) # [B, num_views, num_points, 1]
                # max-pool over views, if mask is 1, otherwise 0
                # align_proj_feats = align_proj_feats.sum(dim=1) / (align_proj_masks.sum(dim=1) + 1e-6) # [B, num_points, F]
                align_proj_feats[~align_proj_masks.expand(-1, -1, -1, align_proj_feats.size(-1))] = -1e6
                align_proj_feats = align_proj_feats.max(dim=1)[0] # [B, num_points, F]
                align_proj_feats[align_proj_feats == -1e6] = 0

                proj_feats = self.proj2d_linear(align_proj_feats)
                
                data_dict["point_clouds"] = data_dict["point_clouds"] + proj_feats

            

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

        if self.use_mae:
            pointgroup_xyz = data_dict["aggregated_vote_xyz"]
            if self.recon_xyz:
                object_feat = torch.cat([pointgroup_xyz, object_feat], dim=-1)
            if data_dict["phase"] == "train":
                loss_mae, object_feat_mae_recon, _ = self.object_mae(object_feat, mask_ratio=self.mask_ratio)
                if self.mae_twice:
                    # re-encode the object feature
                    _, object_feat, _ = self.object_mae(object_feat, encoder_only=True)
                    object_feat = object_feat[:, 1:, :] # remove the [CLS] token
                else:
                    # use the reconstructed object feature
                    object_feat = object_feat_mae_recon
                    if self.recon_xyz:
                        object_feat = object_feat[:, :, 3:]
                data_dict["mae_loss"] = loss_mae
            else:
                # encode the object feature using the MAE encoder
                _, object_feat, _ = self.object_mae(object_feat, encoder_only=True)
                object_feat = object_feat[:, 1:, :] # remove the [CLS] token
                data_dict["mae_loss"] = torch.zeros(1).to(object_feat.device)

            object_feat = self.object_mae_linear(object_feat)
        else:
            data_dict["mae_loss"] = torch.zeros(1).to(object_feat.device)

        if self.use_extra_obj_encoder:
            object_feat = self.object_xencoder(object_feat, object_mask)

        ########################################
        #                                      #
        #             VLM Alignment            #
        #                                      #
        ########################################
        
        if self.use_vlm_align:
            # print(data_dict["bbox_mask"], data_dict["bbox_mask"].shape)
            # align_mask = data_dict["bbox_mask"].bool() # [B, N_p]
            num_proposal = object_feat.size(1)

            # t0 = time()
            object_feat_ts = self.object_aligner(object_feat)  # [B, N_p, C]
            
            # print(f"object-feature-transform: {time() - t0}")

            # --- calculate VLM features
            with torch.no_grad():
                
                scene_ids = data_dict["scene_id_str"]

                if self.random_sample_topk:
                    align_topk = 1
                else:
                    align_topk = self.align_topk
                print(f"Align {align_topk} in {self.align_topk} images")
                
                scene_ids = repeat_elements(scene_ids, align_topk)
                question_ids = data_dict["question_id_str"]
                batch_size = len(question_ids)
                if self.random_sample_topk:
                    image_names = [
                        random.choice(self.scene_view_map[qid][: self.align_topk])
                        for qid in question_ids
                    ]
                else:
                    image_names = sum(
                        [
                            self.scene_view_map[qid][: align_topk]
                            for qid in question_ids
                        ],
                        start=[],
                    )  # [B * K]

                if self.use_blip:
                    # use updated feature
                    # invoke the BLIP model to get the image encoder features
                    # images, poses, depths = self.load_image(data_dict, scene_ids, question_ids, device)
                    image_encoder_feats = image_feats.clone().detach()
                    # FIXME: should this be detached?
                    with torch.no_grad():
                        G = int(image_encoder_feats.size(1) ** 0.5)
                        image_encoder_feats = image_encoder_feats[:, 1:].view(-1, G, G, image_encoder_feats.size(-1)) # remove [CLS], [B, G, G, C]
                        # record cumsum only, for fast mean pooling
                        image_encoder_feats = image_encoder_feats.cumsum(dim=1)
                        image_encoder_feats = image_encoder_feats.cumsum(dim=2)
                        image_encoder_feats = F.pad(image_encoder_feats, (0, 0, 1, 0, 1, 0), "constant", 0) # [B, 1+G, 1+G, C]
                else:
                
                    image_encoder_feats = [
                        self.image_feat_dict[sid][iname]
                        for sid, iname in zip(scene_ids, image_names)
                    ]
                
                image_encoder_feats = torch.stack(image_encoder_feats, dim=0).to(
                    object_feat
                )  # [B * K, 1+G, 1+G, C] or [B * K, 1+G, 1+G, C]
                if self.align_fused_vlm or self.fuse_vlm:
                    image_encoder_feats_recovered = image_encoder_feats.diff(
                        dim=-2
                    ).diff(
                        dim=-3
                    )  # [B * K, G, G, C]
                
                image_names_single = deepcopy(image_names)

                image_names = repeat_elements(
                    image_names, num_proposal
                )  # [B * K * N_p]

                # t0 = time()
                # image_encoder_feats_single = image_encoder_feats.detach().clone()
                
                image_encoder_feats = torch.repeat_interleave(
                    image_encoder_feats, num_proposal, dim=0
                )  # [B * K * N_p, 1+G, 1+G, C]
                nonempty_mask = ~(
                    torch.isnan(image_encoder_feats).all(dim=-1).all(dim=-1).all(dim=-1)
                    > 0
                )  # [B * K * N_p]
                # print(f"repeat image_feats N_propsal times: {time() - t0}")

                if self.use_gt_obj_align:
                    # --- calculate object assignment & objectness label
                    (
                        objectness_label,
                        object_assignment,
                    ) = self.compute_object_assignment(data_dict)
                    # object_assignment [B, N_p], in [0,...,K2-1], K2 ~ number of GT objects

                    gt_bbox = data_dict[
                        "target_bboxes"
                    ]  # [B, K2, 6] (augmented, aligned)
                    gt_corners = calculate_cube_corners(gt_bbox)  # => [B, K2, 8, 3]
                    gt_object_mask = data_dict["box_label_mask"]  # [B, K2]

                    # proposal_assigned_corners[b, i, u, v] = gt_corners[b, obj_assigmnet[b, i, u, v], u, v]
                    corners = torch.gather(
                        input=gt_corners,
                        dim=1,
                        index=object_assignment.unsqueeze(-1)
                        .unsqueeze(-1)
                        .expand(-1, -1, 8, 3),
                    )
                    gt_sem_label = data_dict["sem_cls_label"] # [B, K2]
                    print(gt_sem_label)
                    # import csv
                    # idx2label = {}
                    # label2idx = {}
                    # with open("/home/mowentao/data/ScanQA/data/scannet/meta_data/nyu40_labels.csv", 'r') as file:
                    #     reader = csv.reader(file)
                    #     next(reader)
                    #     for row in reader:
                    #         idx, label = row[:2]
                    #         idx = int(idx)
                    #         idx2label[idx] = label
                    #         label2idx[label] = idx
                    print([DC.class2type[idx.item()] for idx in gt_sem_label[0]])
                    print(scene_ids[0])
                    print(question_ids[0])
                    object_sem_label = torch.gather(
                        input=gt_sem_label,
                        dim=1,
                        index=object_assignment
                    )
                    # gt_object_mask_assgined[b, i] = gt_object_mask[b, obj_assigmnet[b, i]]
                    gt_object_mask_assgined = torch.gather(
                        input=gt_object_mask, dim=1, index=object_assignment
                    )  # [B, N_p]

                    objecness_label = objectness_label.bool()
                    gt_object_mask_assgined  = gt_object_mask_assgined.bool()

                else:
                    corners = data_dict["bbox_corner"].float() # predicted corners

                # --- reverse augment bbox
                if "flip_x" in data_dict:
                    # if saved augmentation info
                    print("reversing augment")
                    corners = reverse_augment(
                        points=corners,  # [B, N_proposal, 8, 3],
                        flip_x=data_dict["flip_x"],
                        flip_y=data_dict["flip_y"],
                        rot_mat=data_dict["rot_mat"],
                        translation=data_dict["translation"],
                        no_rot=False,
                    )

                # if "axis_align_matrix" in data_dict:
                #     print("reversing axis align")
                #     corners = reverse_align(
                #         points=corners,
                #         align_mat=data_dict["axis_align_matrix"],
                #     )

                # if self.jitter_bbox:
                #     corners = jitter_bbox(corners)

                corners = torch.repeat_interleave(
                    corners, align_topk, dim=0
                )  # [B * K, N_p, 8, 3]

                # t0 = time()
                region_feats, overlaps, regions_2d, corner_xy = scene_bbox_to_2d_feat(
                    # bbox_corners_3d=data_dict["bbox_corner"].flatten(0,1),
                    bbox_corners_3d=corners.flatten(0, 1),
                    image_name=image_names,
                    image_feat=image_encoder_feats,
                    grid_size=self.grid_size,
                    device=object_feat.device,
                    image_size=self.image_size,
                )
                # print(region_feats.shape)
                ## region_feats ~ [B * K * N_p, C]
                ## overlaps ~ [B * K * N_p]
                ## regions_2d ~ [B * K * N_p, 4]

                if self.align_one_gt:
                    # compute GT 2d BBOX and 3D proposal 2d BBOX overlap.
                    gt_bbox = data_dict["ref_bbox"] # [B, 6]
                    gt_bbox = torch.tensor(gt_bbox).to(object_feat.device).repeat_interleave(align_topk, dim=0)
                    # [B * K, 6]
                    _, _, gt_regions, _ = scene_bbox_to_2d_feat(
                        bbox_3d=gt_bbox,
                        image_name=image_names_single,
                        image_feat=image_encoder_feats.view(batch_size * align_topk, num_proposal, -1)[:,0,:], # take one repeated 
                        grid_size=self.grid_size,
                        device=object_feat.device,
                        image_size=self.image_size,
                    ) # [B * K, 4]
                    gt_regions = torch.repeat_interleave(gt_regions, num_proposal, dim=0) # [B * K * num_proposal, 4]
                    regions_iou = batch_iou(gt_regions, regions_2d)
                    regions_iou = regions_iou.view(-1, num_proposal)
                    best_region_idx = regions_iou.topk(dim=-1, k=3).indices # [B * K, 3]
                    # best_region_mask = F.one_hot(best_region_idx, num_classes=num_proposal).bool()
                    best_region_mask = multilabel_onehot(best_region_idx, best_region_idx.shape[0], num_proposal).bool().view(-1) # [B * K * N_p]
                    print(best_region_mask.shape, overlaps.shape)
                    overlaps[~best_region_mask] = 0 # remove non-best regions in aligning by setting overlap to 0

                    
                overlaps = overlaps.view(
                    -1, align_topk, num_proposal
                )  # [B, K, N_p]
                align_mask = data_dict["bbox_mask"]
                if self.use_gt_obj_align:
                    # align_mask = align_mask & objectness_label & gt_object_mask_assgined
                    align_mask = align_mask & gt_object_mask_assgined
                align_mask = align_mask.unsqueeze(1).expand(
                    -1, align_topk, -1
                )  # [B, N_p] => [B, K, N_p]
                align_mask = align_mask & (
                    (overlaps > self.overlap_threshold).to(align_mask)
                )  # [B, K, N_p]
                align_mask = align_mask & (
                    nonempty_mask.view(-1, align_topk, num_proposal)
                ) # [B, K, N_p]

                region_feats = region_feats.view(
                    -1, align_topk, num_proposal, region_feats.shape[-1]
                )  # [B, K, N_p, C]
                regsum = torch.sum(align_mask.unsqueeze(-1), dim=1)  # [B, N_p, 1]

                if self.use_vs:
                    with torch.set_grad_enabled(True):
                        region_feats = self.forward_vs(region_feats, align_mask) # => [B, N_p, C]
                else:
                    # print(region_feats.shape, align_mask.shape)
                    region_feats = (
                        (region_feats * align_mask.unsqueeze(-1)).sum(dim=1)
                    ) / regsum  # [B, N_p, C]

                # print(f"get region_feats: {time() - t0}")
                ## region_feats = torch.stack(region_feats, dim=0).to(object_feat) # [B * N_p, C]
                ## region_feats = region_feats.view(-1, num_proposal, region_feats.size(-1)) # [B, N_p, C]
                # overlap_mask = (overlaps.view(-1, num_proposal) > self.overlap_threshold).to(align_mask)
                # align_mask = data_dict["bbox_mask"].bool() & overlap_mask
                align_mask = regsum.squeeze(-1) >= 1 # [B, N_p]
                
                print(
                    f"alignining {align_mask.sum(dim=-1).cpu().tolist()} objects (proposal)"
                )
                # print(region_feats.shape, align_mask.shape)

                # save latest bboxes for debug
                # torch.save(corners, "/home/mowentao/data/ScanQA/debug/latest-corners.pkl")
                # torch.save(data_dict["point_clouds"], "/home/mowentao/data/ScanQA/debug/pcs.pkl")
                # TODO: print Image, print 3D BBOX, print 2D BBOX
                if data_dict["phase"] == "train" and self.visualize_bbox and align_mask.sum() > 0:
                    from PIL import Image
                    import matplotlib.pyplot as plt
                    import matplotlib.patches as patches
                    from matplotlib import collections as mc

                    import os
                    fig, ax = plt.subplots()
                    iid = align_mask.sum(dim=-1).nonzero()[0].item()
                    iname = image_names_single[iid]
                    sid = scene_ids[iid]
                    ipath = os.path.join(DSET_VIEWS_PATH, sid, iname)
                    print(ipath)
                    img = plt.imread(ipath)
                    ax.imshow(img)
                    try:
                        oid = align_mask[iid].nonzero()[0].item()
                        bbox = regions_2d.view(batch_size, align_topk, num_proposal, -1)[iid, 0, oid].detach().cpu().tolist()
                        bbox_3d_proj = corner_xy.reshape(batch_size, align_topk, num_proposal, 8, 2)[iid, 0, oid].detach().cpu().tolist()
                        bbox_sem_label = object_sem_label.reshape(batch_size, align_topk, num_proposal)[iid, 0, oid].item()
                        print(oid, bbox, bbox_3d_proj)
                        print(corners[iid, oid])
                        # print(regions_2d[oid]) # xmin xmax ymin ymax
                        rect = patches.Rectangle(
                            (bbox[0]*img.shape[1], bbox[2]*img.shape[0]), 
                            bbox[1]*img.shape[1] - bbox[0]*img.shape[1], 
                            bbox[3]*img.shape[0] - bbox[2]*img.shape[0], 
                            linewidth=2, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)

                        # import csv
                        # idx2label = {}
                        # label2idx = {}
                        # with open("/home/mowentao/data/ScanQA/data/scannet/meta_data/nyu40_labels.csv", 'r') as file:
                        #     reader = csv.reader(file)
                        #     next(reader)
                        #     for row in reader:
                        #         idx, label = row[:2]
                        #         idx = int(idx)
                        #         idx2label[idx] = label
                        #         label2idx[label] = idx

                        object_index = object_assignment.reshape(batch_size, align_topk, num_proposal)[iid, 0, oid].item()

                        ax.text(bbox[0]*img.shape[1], bbox[2]*img.shape[0], f"{object_index}: {DC.class2type[bbox_sem_label]}", 
                            color='white', fontsize=12, fontweight='bold', 
                            bbox=dict(facecolor='red', alpha=0.8, edgecolor='none')
                        )
                        
                        lc_3d = mc.LineCollection(
                            corners_to_edges(bbox_3d_proj),
                            linewidths=2,
                            colors=(0,0,1,1)
                        )               
                        ax.add_collection(lc_3d)
                    except IndexError as e:
                        # no any alignable object
                        pass
                    plt.show()
                    # pause()

            data_dict["region_feats"] = region_feats
            data_dict["align_mask"] = align_mask
            data_dict["align_bs"] = object_feat.size(0)
            data_dict["object_feat_ts"] = object_feat_ts

            
            if self.use_variational_aligner:
                if self.use_separate_vae:
                    region_feats_recon, region_mu, region_logvar = self.region_vae(region_feats[align_mask])
                else:
                    region_feats_recon, region_mu, region_logvar = self.object_vae(region_feats[align_mask])
                object_feats_recon, object_mu, object_logvar = self.object_vae(object_feat_ts[align_mask])
                data_dict["region_feats_recon"] = region_feats_recon
                data_dict["region_mu"] = region_mu
                data_dict["region_logvar"] = region_logvar
                data_dict["object_feats_recon"] = object_feats_recon
                data_dict["object_mu"] = object_mu
                data_dict["object_logvar"] = object_logvar
                
            if self.replace_3d_feature:
                ic(object_feat.shape, region_feats.shape, align_mask.shape, object_mask.shape)
                object_feat = torch.where(
                    align_mask.unsqueeze(-1),
                    self.object_aligner_rev(region_feats),
                    object_feat,
                )
                align_mask_ = align_mask.unsqueeze(1).unsqueeze(2)
                object_mask = torch.where(
                    align_mask_,
                    align_mask_,
                    object_mask,
                )


            # --- calculate blip fused loss
            if self.align_fused_vlm or self.fuse_vlm:
                image_encoder_feats_recovered = self.blip_feat_linear(
                    image_encoder_feats_recovered
                ).flatten(-3, -2)
                # => [B * K, G**2, C]
                blip_mask = (
                    torch.zeros(
                        image_encoder_feats_recovered.shape[:-1],
                        device=image_encoder_feats_recovered.device,
                    )
                    .bool()
                    .unsqueeze(1)
                    .unsqueeze(2)
                )
                blip_lang_feat = torch.repeat_interleave(
                    lang_feat, self.align_topk, dim=0
                )
                blip_lang_mask = torch.repeat_interleave(
                    lang_mask, self.align_topk, dim=0
                )
                print(image_encoder_feats_recovered.shape, blip_mask.shape)
                print(object_feat.shape, object_mask.shape)

                blip_lang_feat, blip_grid_feat = self.blip_fusion_backbone(
                    blip_lang_feat,
                    image_encoder_feats_recovered,
                    blip_lang_mask,
                    blip_mask,
                )
                blip_lang_feat = self.blip_attflat_lang(blip_lang_feat, blip_lang_mask)
                blip_grid_feat = self.blip_attflat_visual(
                    blip_grid_feat, 
                    blip_mask, 
                    self.att_pdrop if data_dict["phase"] == "train" else 0,
                    self.att_drop_topk,
                )
                blip_fused_feat = self.fusion_norm(
                    blip_lang_feat + blip_grid_feat
                ).view(-1, self.align_topk, blip_lang_feat.shape[-1])
                # [B, K, mcan_flat_out_size]
                if self.align_fused_vlm:
                    # record for loss computation
                    data_dict["blip_fused_feat"] = blip_fused_feat
                if self.selector_head is not None:
                    data_dict["selector_score"] = self.selector_head(blip_fused_feat)

        # --- QA
        if not self.use_blip:
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

            if self.fuse_vlm:
                fuse_feat = self.fusion_norm(fuse_feat + blip_fused_feat.mean(dim=1))

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
                elif self.scene_feature_type == "locsem":
                    (
                        objectness_label,
                        object_assignment,
                    ) = self.compute_object_assignment(data_dict)
                    sem_cls_label = torch.gather(data_dict['sem_cls_label'], 1, object_assignment)
                    target_bboxes = torch.gather(data_dict['target_bboxes'], 1, object_assignment.unsqueeze(-1).repeat(1,1,6)) # [B, N_bbox, 6] => [B, N_proposal, 6]
                    box_label_mask = torch.gather(data_dict['box_label_mask'], 1, object_assignment) # [B, N_bbox] => [B, N_proposal], 1 for valid, 0 for invalid
                    object_mask_for_2d = box_label_mask.to(object_mask) # [B, N_proposal]
                    object_feat_for_2d = torch.cat([target_bboxes, sem_cls_label.unsqueeze(-1)], dim=-1) # [B, N_proposal, 6+1]
                elif self.scene_feature_type == "locsemfeat":
                    (
                        objectness_label,
                        object_assignment,
                    ) = self.compute_object_assignment(data_dict)
                    sem_cls_label = torch.gather(data_dict['sem_cls_label'], 1, object_assignment)
                    target_bboxes = torch.gather(data_dict['target_bboxes'], 1, object_assignment.unsqueeze(-1).repeat(1,1,6)) # [B, N_bbox, 6] => [B, N_proposal, 6]
                    box_label_mask = torch.gather(data_dict['box_label_mask'], 1, object_assignment) # [B, N_bbox] => [B, N_proposal], 1 for valid, 0 for invalid
                    object_mask_for_2d = box_label_mask.to(object_mask) # [B, N_proposal]
                    object_mask_for_2d = object_mask_for_2d & (~object_mask.squeeze(1).squeeze(1)) # for BLIP/Transformers, 1 for valid, 0 for invalid
                    object_feat_for_2d = torch.cat([target_bboxes, sem_cls_label.unsqueeze(-1)], dim=-1) # [B, N_proposal, 6+1]
                    object_feat_for_2d = torch.cat([object_feat_for_2d, object_feat], dim=-1) # [B, N_proposal, 6+1+hidden_size]
                else:
                    raise NotImplementedError

                depth_map = None
                if self.depth_fusion:
                    depth_map = enet_to_blip(depths[:, 0].unsqueeze(1)) # [B, 1, H, W]
                    depth_map = depth_map.squeeze(1).flatten(-2, -1).unsqueeze(-1) # [B, H*W, 1]

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
