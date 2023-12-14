import os
import sys
import json
import argparse
import collections
import torch
import torch.optim as optim
import numpy as np
import wandb
import logging

from torch.utils.data import DataLoader
from datetime import datetime

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
# sys.path.append("/home/mowentao/scratch/BLIP") # HACK add BLIP folder
from lib.dataset import ScannetQADataset, ScannetQADatasetConfig
from lib.solver import Solver
from lib.config import CONF 
from models.qa_module import ScanQA

from utils.vlm_align_util import *

from pprint import pprint
import pretty_errors
from icecream import ic
from argparse import Namespace

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

project_name = "ScanQA_v1.0"
SCANQA_TRAIN = json.load(open(os.path.join(CONF.PATH.SCANQA, project_name + "_train.json"))) 
SCANQA_VAL = json.load(open(os.path.join(CONF.PATH.SCANQA, project_name + "_val.json")))

# constants
DC = ScannetQADatasetConfig()
GLOBAL = Namespace()

logger = logging.getLogger(__name__)

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="debugging mode")
    parser.add_argument("--tag", type=str, help="tag for the training, e.g. XYZ_COLOR", default="")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    # Training
    parser.add_argument("--cur_criterion", type=str, default="answer_acc_at1", help="data augmentation type")
    parser.add_argument("--batch_size", type=int, help="batch size", default=16)
    parser.add_argument("--train_batch_size", type=int, help="batch size", default=16)
    parser.add_argument("--val_batch_size", type=int, help="batch size", default=16)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=50)
    parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=10)
    parser.add_argument("--val_step", type=int, help="iterations of validating", default=1000) # 5000
    parser.add_argument("--train_num_scenes", type=int, default=-1, help="Number of train scenes [default: -1]")
    parser.add_argument("--val_num_scenes", type=int, default=-1, help="Number of val scenes [default: -1]")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    # Optimizer   
    parser.add_argument("--optim_name", type=str, help="optimizer name", default="adam")
    parser.add_argument("--wd", type=float, help="weight decay", default=1e-5)
    parser.add_argument("--lr", type=float, help="initial learning rate", default=5e-4)
    parser.add_argument("--adam_beta1", type=float, help="beta1 hyperparameter for the Adam optimizer", default=0.9)
    parser.add_argument("--adam_beta2", type=float, help="beta2 hyperparameter for the Adam optimizer", default=0.999) # 0.98
    parser.add_argument("--adam_epsilon", type=float, help="epsilon hyperparameter for the Adam optimizer", default=1e-8) # 1e-9
    parser.add_argument("--amsgrad", action="store_true", help="Use amsgrad for Adam")
    parser.add_argument('--lr_decay_step', nargs='+', type=int, default=[100, 200]) # 15
    parser.add_argument("--lr_decay_step_2d", nargs='+', type=int, default=[100, 200]) # 01, 0.2
    parser.add_argument("--lr_decay_rate", type=float, help="decay rate of learning rate", default=0.2) # 01, 0.2
    parser.add_argument('--bn_decay_step', type=int, default=20)
    parser.add_argument("--bn_decay_rate", type=float, help="bn rate", default=0.5)
    parser.add_argument("--max_grad_norm", type=float, help="Maximum gradient norm ", default=1.0)
    parser.add_argument("--scheduler_type", type=str, default="step")
    # Data
    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_augment", action="store_true", help="Do NOT use data augmentations.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    # Model
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden layer size[default: 256]")
    ## pointnet & votenet & proposal
    parser.add_argument("--vote_radius", type=float, help="", default=0.3) # 5
    parser.add_argument("--vote_nsample", type=int, help="", default=16) # 512
    parser.add_argument("--pointnet_width", type=int, help="", default=1)
    parser.add_argument("--pointnet_depth", type=int, help="", default=2)
    parser.add_argument("--seed_feat_dim", type=int, help="", default=256) # or 288
    parser.add_argument("--proposal_size", type=int, help="", default=128)    
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--use_seed_lang", action="store_true", help="Fuse seed feature and language feature.")    
    ## module option
    parser.add_argument("--no_object_mask", action="store_true", help="objectness_mask for qa")
    parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
    parser.add_argument("--no_answer", action="store_true", help="Do NOT train the localization module.")
    parser.add_argument("--no_detection", action="store_true", help="Do NOT train the detection module.")
    parser.add_argument("--no_reference", action="store_true", help="Do NOT train the localization module.")
    # Pretrain
    parser.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")
    # Loss
    parser.add_argument("--vote_loss_weight", type=float, help="vote_net loss weight", default=1.0)
    parser.add_argument("--objectness_loss_weight", type=float, help="objectness loss weight", default=0.5)
    parser.add_argument("--box_loss_weight", type=float, help="box loss weight", default=1.0)
    parser.add_argument("--sem_cls_loss_weight", type=float, help="sem_cls loss weight", default=0.1)
    parser.add_argument("--ref_loss_weight", type=float, help="reference loss weight", default=0.1)
    parser.add_argument("--lang_loss_weight", type=float, help="language loss weight", default=0.1)
    parser.add_argument("--answer_loss_weight", type=float, help="answer loss weight", default=0.1)  
    # Answer
    parser.add_argument("--answer_cls_loss", type=str, help="answer classifier loss", default="bce") # ce, bce
    parser.add_argument("--answer_max_size", type=int, help="maximum size of answer candicates", default=-1) # default use all
    parser.add_argument("--answer_min_freq", type=int, help="minimum frequence of answers", default=1)
    parser.add_argument("--answer_pdrop", type=float, help="dropout_rate of answer_cls", default=0.3)
    # Question
    parser.add_argument("--tokenizer_name", type=str, help="Pretrained tokenizer name", default="spacy_blank") # or bert-base-uncased, bert-large-uncased-whole-word-masking, distilbert-base-uncased
    parser.add_argument("--lang_num_layers", type=int, default=1, help="Number of GRU layers")
    parser.add_argument("--lang_use_bidir", action="store_true", help="Use bi-directional GRU.")
    parser.add_argument("--freeze_bert", action="store_true", help="Freeze BERT ebmedding model")
    parser.add_argument("--finetune_bert_last_layer", action="store_true", help="Finetue BERT last layer")
    parser.add_argument("--lang_pdrop", type=float, help="dropout_rate of lang_cls", default=0.3)
    ## MCAN
    parser.add_argument("--mcan_pdrop", type=float, help="", default=0.1)
    parser.add_argument("--mcan_flat_mlp_size", type=int, help="", default=256) # mcan: 512
    parser.add_argument("--mcan_flat_glimpses", type=int, help="", default=1)
    parser.add_argument("--mcan_flat_out_size", type=int, help="", default=512) # mcan: 1024
    parser.add_argument("--mcan_num_heads", type=int, help="", default=8)
    parser.add_argument("--mcan_num_layers", type=int, help="", default=2) # mcan: 6
    # VLM-align
    parser.add_argument("--vlm_hidden_size", type=int, default=768, help="hidden size of vlm to be aligned") # vit-base: 1024
    parser.add_argument("--use_vlm_align", action="store_true", help="Use VLM Align")
    parser.add_argument("--overlap_threshold", type=float, default=0.7, help="") 
    parser.add_argument("--i2tfile", type=str, default="/home/mowentao/scratch/BLIP/scene_eval_new.json")
    parser.add_argument("--i2tfile_eval", type=str, default="")
    parser.add_argument("--objectness_threshold", type=float, default=0.3)
    parser.add_argument("--align_loss_weight", type=float, default=0.3)
    parser.add_argument("--align_topk", type=int, default=1)
    parser.add_argument("--begin_align_epoch", type=int, default=0)
    parser.add_argument("--align_fused_vlm", action="store_true", help="")
    parser.add_argument("--fuse_vlm", action="store_true", help="")
    parser.add_argument("--use_gt_obj_align", action="store_true", help="")
    parser.add_argument("--random_sample_topk", action="store_true", help="")
    parser.add_argument("--simple_align", action="store_true", help="")
    parser.add_argument("--use_extra_obj_encoder", action="store_true", help="")
    parser.add_argument("--use_contrastive", action="store_true", help="")
    parser.add_argument("--contrastive_temperature", type=float, default=10)
    parser.add_argument("--use_vs", action="store_true", help="")
    # Hard-positive mining
    parser.add_argument("--jitter_bbox", action="store_true", help="Jitter BBox")
    parser.add_argument("--att_pdrop", type=float, default=0)
    parser.add_argument("--att_drop_topk", type=int, default=100)
    parser.add_argument("--use_variational_aligner", action="store_true")
    parser.add_argument("--use_separate_vae", action="store_true")
    parser.add_argument("--vae_latent_size", type=int, default=128)
    parser.add_argument("--save_pred", action="store_true")
    parser.add_argument("--align_one_gt", action="store_true")
    parser.add_argument("--use_soft_label_align", action="store_true")
    parser.add_argument("--soft_label_path", type=str, default="/home/mowentao/scratch/BLIP/2d_finetuned_pred_gpt3.5_trainval.json")
    parser.add_argument("--alternative_ckpt", type=str, default="")
    parser.add_argument("--visualize_bbox", action="store_true")
    parser.add_argument("--image_size", type=int, default=768, help="image size of rendered views")
    parser.add_argument("--replace_3d_feature", action="store_true")
    

    # CLIP/BLIP 
    parser.add_argument("--image_encoder_type", type=str, default="blip", help="clip or blip")
    parser.add_argument("--clip_model_name", type=str, default="ViT-B-16", help="clip model name")
    parser.add_argument("--clip_ckpt_name", type=str, default="laion2b_s34b_b88k", help="clip pretrained path")
    parser.add_argument("--clip_return_layer", type=int, default=-1, help="clip return layer")
    parser.add_argument("--use_clip_lang", action="store_true", help="use clip language encoder")
    parser.add_argument("--clip_lr_scale", type=float, default=0.05, help="clip lr scale")

    # PointNet MAE
    parser.add_argument("--use_mae", action="store_true")
    parser.add_argument("--mae_loss_weight", type=float, default=0.3)
    parser.add_argument("--mae_mask_ratio", type=float, default=0.75)
    parser.add_argument("--recon_xyz", action="store_true")
    
    # CapQA
    parser.add_argument("--sideload_qa", action="store_true")
    parser.add_argument("--sideload_append", action="store_true")
    ## DDP option
    parser.add_argument("--ddp", action="store_true", help="Use DDP.")
    ## BLIP integration
    parser.add_argument("--use_blip", action="store_true", help="Use BLIP.")
    # parser.add_argument("--i2tfile", type=str, default="/home/mowentao/scratch/BLIP/scene_eval_new.json", help="scene-image mapping file")
    parser.add_argument("--lr_blip", type=float, default=1e-4, help="lr for blip")
    parser.add_argument("--lr_blip3d", type=float, default=1e-4, help="lr for blip3d")
    parser.add_argument("--wd_blip", type=float, default=0, help="weight decay for blip")
    parser.add_argument("--use_text_decoder", action="store_true", help="Use text decoder.")
    parser.add_argument("--dset_views_path", type=str, default="", help="path to scene views")
    parser.add_argument("--scene_feature_position", type=str, default="image", help="where to put scene feature")
    parser.add_argument("--scene_feature_type", type=str, default="full", help="which type of scene feature to fuse")
    parser.add_argument("--use_scene_weight", action="store_true", help="Use scene weight.")
    parser.add_argument("--project_2d", action="store_true", help="Project 2D feature to 3D.")
    parser.add_argument("--depth_fusion", action="store_true", help="Fuse depth.")
    parser.add_argument("--use_scene_classifier", action="store_true", help="Use scene classifier.")
    parser.add_argument("--med_config", type=str, default="/home/mowentao/scratch/BLIP/configs/med_config.json")
    parser.add_argument("--use_scene_classifier_2d3d", action="store_true", help="Use scene classifier-2d3d.")
    parser.add_argument("--not_copy_weights", action="store_true", help="Not copy weights from 2D to 3D.")
    parser.add_argument("--scene_encoder_layers", type=int, default=-1, help="scene encoder layers")
    parser.add_argument("--mix_tokens", action="store_true", help="mix tokens")
    parser.add_argument("--share_decoder", action="store_true")
    parser.add_argument("--num_hidden_layers_twin", type=int, default=None) # if none, use num_hidden_layers
    parser.add_argument("--random_scene_view", action="store_true")
    ## Pretrained VoteNet 
    parser.add_argument("--votenet_ckpt", type=str, default="", help="Pretrained VoteNet checkpoint")
    ## Two stage training
    parser.add_argument("--stage", type=str, default="", help="stage of training (DET or QA), if empty, no two stage training")
    parser.add_argument("--first_stage_ckpt_path", type=str, default="", help="Pretrained first stage checkpoint")
    ## No scene
    parser.add_argument("--no_scene", action="store_true", help="No scene.")
    
    ## Reinit
    parser.add_argument("--reinit_epoch", type=int, default=-1, help="Reinit.")

    ### GRL
    parser.add_argument("--grl", action="store_true", help="Use GRL.")
    parser.add_argument("--grl_weight", type=float, default=0.1, help="GRL loss weight.")

    ### Customize encoder/decoder layers if "replace" (i.e. 3D only)
    parser.add_argument("--encoder_layers", type=int, default=None)
    parser.add_argument("--decoder_layers", type=int, default=None)
    parser.add_argument("--random_init_blip", action="store_true")

    ### Use ViLT as VLM
    parser.add_argument("--use_vilt", action="store_true")
    
    args = parser.parse_args()
    pprint(args)
    return args
    
def init_distributed_mode(args):
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    args.world_size = torch.distributed.get_world_size()
    args.local_rank = torch.distributed.get_rank()
    # args.device = torch.device("cuda", args.local_rank)
    torch.cuda.set_device(args.local_rank)
    return args

def get_answer_cands(args, scanqa):
    answer_counter = sum([data["answers"] for data in scanqa["train"]], [])
    answer_counter += sum([data["answers"] for data in scanqa["val"]], []) 
    
    answer_counter = collections.Counter(sorted(answer_counter))
    num_all_answers = len(answer_counter)
    answer_max_size = args.answer_max_size
    if answer_max_size < 0:
        answer_max_size = len(answer_counter)
    answer_counter = dict([x for x in answer_counter.most_common()[:answer_max_size] if x[1] >= args.answer_min_freq])
    print("using {} answers out of {} ones".format(len(answer_counter), num_all_answers))    
    answer_cands = sorted(answer_counter.keys(), key=lambda x: (-answer_counter[x], x)) # sort by frequency
    return answer_cands, answer_counter


def get_dataloader(args, scanqa, all_scene_list, split, config, augment, batch_size):
    try: 
        answer_cands = GLOBAL.answer_cands
        answer_counter = GLOBAL.answer_counter
    except:
        answer_cands, answer_counter = get_answer_cands(args, scanqa)
        GLOBAL.answer_cands = answer_cands
        GLOBAL.answer_counter = answer_counter
    config.num_answers = len(answer_cands)

    if 'bert-' in args.tokenizer_name or 't5-' in args.tokenizer_name: 
        from transformers import AutoTokenizer
        # os.environ["TOKENIZERS_PARALLELISM"] = "true" # NOTE: comment this on first download
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    else:
        tokenizer = None

    if args.sideload_qa and split == "train":
        sideload_qa_path = "/home/mowentao/scratch/BLIP/fixed_qa.json"
    else:
        sideload_qa_path = None

    dataset = ScannetQADataset(
        scanqa=scanqa[split], 
        scanqa_all_scene=all_scene_list, 
        answer_cands=answer_cands,
        answer_counter=answer_counter,
        answer_cls_loss=args.answer_cls_loss,
        split=split, 
        num_points=args.num_points, 
        use_height=(not args.no_height),
        use_color=args.use_color, 
        use_normal=args.use_normal, 
        use_multiview=args.use_multiview,
        tokenizer=tokenizer,
        augment=augment,
        debug=args.debug,
        sideload_qa_path=sideload_qa_path,
        sideload_append=args.sideload_append and split == "train",
        i2tfile=args.i2tfile if split == "train" or len(args.i2tfile_eval) == 0 else args.i2tfile_eval,
        scene_view_topk=args.align_topk,
        dset_views_path=args.dset_views_path,
        random_scene_view=args.random_scene_view,
    )
    if args.ddp:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=(split == "train"), seed=args.seed)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, sampler=sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=8)
    return dataset, dataloader


def get_model(args, config):
    if "bert-" in args.tokenizer_name or "t5-" in args.tokenizer_name:
        from transformers import AutoConfig
        bert_model_name = args.tokenizer_name
        bert_config = AutoConfig.from_pretrained(bert_model_name)
        if "t5-" in bert_model_name:
            lang_emb_size = bert_config.d_model
        if hasattr(bert_config, "hidden_size"):
            lang_emb_size = bert_config.hidden_size
        else:
            # for distllbert
            lang_emb_size = bert_config.dim
    else:
        bert_model_name = None
        lang_emb_size = 300 # glove emb_size

    # initiate model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)

    model = ScanQA(
        num_answers=config.num_answers,
        # proposal
        input_feature_dim=input_channels,            
        num_object_class=config.num_class, 
        num_heading_bin=config.num_heading_bin,
        num_size_cluster=config.num_size_cluster,
        mean_size_arr=config.mean_size_arr,
        num_proposal=args.num_proposals, 
        seed_feat_dim=args.seed_feat_dim,
        proposal_size=args.proposal_size,
        pointnet_width=args.pointnet_width,
        pointnet_depth=args.pointnet_depth,        
        vote_radius=args.vote_radius, 
        vote_nsample=args.vote_nsample,            
        # qa
        #answer_cls_loss="ce",
        answer_pdrop=args.answer_pdrop,
        mcan_num_layers=args.mcan_num_layers,
        mcan_num_heads=args.mcan_num_heads,
        mcan_pdrop=args.mcan_pdrop,
        mcan_flat_mlp_size=args.mcan_flat_mlp_size, 
        mcan_flat_glimpses=args.mcan_flat_glimpses,
        mcan_flat_out_size=args.mcan_flat_out_size,
        # lang
        lang_use_bidir=args.lang_use_bidir,
        lang_num_layers=args.lang_num_layers,
        lang_emb_size=lang_emb_size,
        lang_pdrop=args.lang_pdrop,
        bert_model_name=bert_model_name,
        freeze_bert=args.freeze_bert,
        finetune_bert_last_layer=args.finetune_bert_last_layer,
        # common
        hidden_size=args.hidden_size,
        # option
        use_object_mask=(not args.no_object_mask),
        use_lang_cls=(not args.no_lang_cls),
        use_reference=(not args.no_reference),
        use_answer=(not args.no_answer),      
        # vlm align
        use_vlm_align=args.use_vlm_align,
        replace_3d_feature=args.replace_3d_feature,
        vlm_hidden_size=args.vlm_hidden_size,
        image_feat_dict=args.image_feat_dict,
        grid_size=args.grid_size,
        scene_view_map=args.scene_view_map,
        align_fused_vlm=args.align_fused_vlm,
        fuse_vlm=args.fuse_vlm,
        use_gt_obj_align=args.use_gt_obj_align,
        # bbox_data=args.bbox_data,
        objectness_threshold=args.objectness_threshold,
        overlap_threshold=args.overlap_threshold,
        align_topk=args.align_topk,
        random_sample_topk=args.random_sample_topk,
        simple_align=args.simple_align,
        use_extra_obj_encoder=args.use_extra_obj_encoder,
        use_contrastive=args.use_contrastive,
        contrastive_temperature=args.contrastive_temperature,
        use_variational_aligner=args.use_variational_aligner,
        vae_latent_size=args.vae_latent_size,
        use_separate_vae=args.use_separate_vae,
        use_vs=args.use_vs,
        # hard-positive mining
        jitter_bbox=args.jitter_bbox,
        att_pdrop=args.att_pdrop, 
        att_drop_topk=args.att_drop_topk,
        align_one_gt=args.align_one_gt,
        use_soft_label_align=args.use_soft_label_align,
        soft_label_on_train=True,
        soft_label_path=args.soft_label_path,
        visualize_bbox=args.visualize_bbox,
        image_size=args.image_size,
        use_mae=args.use_mae,
        mask_ratio=args.mae_mask_ratio,
        recon_xyz=args.recon_xyz,
        use_clip_lang=args.use_clip_lang,
        clip_model_name=args.clip_model_name,
        clip_ckpt_name=args.clip_ckpt_name,
        # save_pred=args.save_pred
        use_blip=args.use_blip,
        votenet_ckpt=args.votenet_ckpt,
        use_text_decoder=args.use_text_decoder,
        all_answers=GLOBAL.answer_cands,
        stage=args.stage,
        dset_views_path=args.dset_views_path,
        scene_feature_position=args.scene_feature_position,
        scene_feature_type=args.scene_feature_type,
        use_scene_weight=args.use_scene_weight,
        i2tfile=args.i2tfile,
        i2tfile_eval=args.i2tfile_eval if len(args.i2tfile_eval) > 0 else args.i2tfile,
        no_scene=args.no_scene,
        project_2d=args.project_2d,
        first_stage_ckpt_path=args.first_stage_ckpt_path,
        depth_fusion=args.depth_fusion,
        use_scene_classifier=args.use_scene_classifier,
        med_config=args.med_config,
        use_scene_classifier_2d3d=args.use_scene_classifier_2d3d,
        not_copy_weights=args.not_copy_weights,
        scene_encoder_layers=args.scene_encoder_layers,
        mix_tokens=args.mix_tokens,
        share_decoder=args.share_decoder,
        num_hidden_layers_twin=args.num_hidden_layers_twin,
        grl=args.grl,
        grl_weight=args.grl_weight,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        random_init_blip=args.random_init_blip,
        use_vilt=args.use_vilt,
    )

    model.load_pretrained()
    # load first stage checkpoint
    # if args.stage == "VQA" and args.first_stage_ckpt_path != "":
    #     model_dict = model.state_dict()
    #     pretrained_dict = torch.load(args.first_stage_ckpt_path, map_location="cpu")
    #     torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(pretrained_dict, "module.")

    #     # 1. filter out unnecessary keys
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #     print(list(pretrained_dict.keys()))
    #     # 2. overwrite entries in the existing state dict
    #     model_dict.update(pretrained_dict)
    #     # 3. load the new state dict
    #     model.load_state_dict(model_dict)

    # to CUDA
    model = model.to(torch.device("cuda", args.local_rank))
    print(next(model.parameters()).device)

    

    if args.ddp:
        model = DDP(model, find_unused_parameters=True)
    return model


def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params

def get_solver(args, dataloader):
    model = get_model(args, DC)
    #wandb.watch(model, log_freq=100)

    if args.optim_name == 'adam':
        model_params = [{"params": model.parameters()}]
        optimizer = optim.Adam(
            model_params,
            lr=args.lr, 
            betas=[args.adam_beta1, args.adam_beta2],
            eps=args.adam_epsilon,
            weight_decay=args.wd, 
            amsgrad=args.amsgrad)
    elif args.optim_name == 'adamw':
        if args.use_clip_lang:
            p_lang = []
            p_other = []
            for name, param in model.named_parameters():
                if 'lang_net' in name:
                    p_lang.append(param)
                else:
                    p_other.append(param)
            param_groups = [{'params': p_lang, 'lr': args.lr * args.clip_lr_scale},
                            {'params': p_other}]
            optimizer = optim.AdamW(param_groups, lr=args.lr, 
                                    betas=[args.adam_beta1, args.adam_beta2],
                                    eps=args.adam_epsilon,
                                    weight_decay=args.wd, 
                                    amsgrad=args.amsgrad)
        elif args.use_blip:
            p_blip = []
            p_other = []
            p_blip3d = []
            for name, param in model.named_parameters():
                if 'blip_model' in name :
                    if 'layer_twin' in name:
                        # Bert-TWIN layer
                        p_blip3d.append(param)
                    elif 'text_encoder_scene' in name \
                            or 'text_decoder_scene' in name:
                        p_blip3d.append(param)
                    elif 'lowrank' in name \
                            or 'fusion' in name \
                            or 'gated' in name \
                            or 'lang_net' in name \
                            or 'answer_cls' in name \
                            or 'linear_scene_object' in name \
                            or 'scene_weight' in name \
                            or 'classifier' in name:
                        p_other.append(param)
                    elif args.lr_blip > 1e-9:
                        p_blip.append(param)
                else:
                    p_other.append(param)
            param_groups = [{'params': p_blip, 'lr': args.lr_blip, 'weight_decay': args.wd_blip},
                            {'params': p_blip3d, 'lr': args.lr_blip3d, 'weight_decay': args.wd_blip},
                            {'params': p_other}]
            optimizer = optim.AdamW(param_groups, lr=args.lr, 
                                    betas=[args.adam_beta1, args.adam_beta2],
                                    eps=args.adam_epsilon,
                                    weight_decay=args.wd, 
                                    amsgrad=args.amsgrad)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, 
                                    betas=[args.adam_beta1, args.adam_beta2],
                                    eps=args.adam_epsilon,
                                    weight_decay=args.wd, 
                                    amsgrad=args.amsgrad)
    elif args.optim_name == 'adamw_cb':
        from transformers import AdamW
        optimizer = AdamW(model.parameters(), lr=args.lr, 
                                betas=[args.adam_beta1, args.adam_beta2],
                                eps=args.adam_epsilon,
                                weight_decay=args.wd)
    else:
        raise NotImplementedError

    total_steps = args.epoch * len(dataloader["train"])
    print(f"total {total_steps} train iters")
    
    # if args.scheduler_name == "":
    #     scheduler = None
    # elif args.scheduler_name == "linear":
    #     scheduler = optim.lr_scheduler.LinearLR(optimizer, 1, 0.01,)

    print('set optimizer...')
    print(optimizer)
    # print(scheduler)
    print()
    

    if args.use_checkpoint:
        print("loading checkpoint {}...".format(args.use_checkpoint))
        stamp = args.use_checkpoint
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        checkpoint = torch.load(os.path.join(CONF.PATH.OUTPUT, args.use_checkpoint, "checkpoint.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.tag: stamp += "_"+args.tag.upper()
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        os.makedirs(root, exist_ok=True)

    loss_weights = {}
    loss_weights['vote_loss']       = args.vote_loss_weight
    loss_weights['objectness_loss'] = args.objectness_loss_weight 
    loss_weights['box_loss']        = args.box_loss_weight
    loss_weights['sem_cls_loss']    = args.sem_cls_loss_weight
    loss_weights['ref_loss']        = args.ref_loss_weight
    loss_weights['lang_loss']       = args.lang_loss_weight
    loss_weights['answer_loss']     = args.answer_loss_weight
    loss_weights['align_loss']      = args.align_loss_weight
    loss_weights['mae_loss']        = args.mae_loss_weight

    solver = Solver(
        model=model, 
        config=DC, 
        dataloader=dataloader, 
        optimizer=optimizer, 
        stamp=stamp, 
        val_step=args.val_step,
        epoch=args.epoch,
        cur_criterion=args.cur_criterion,
        detection=not args.no_detection,
        use_reference=not args.no_reference, 
        use_answer=not args.no_answer,
        use_lang_classifier=not args.no_lang_cls,
        max_grad_norm=args.max_grad_norm,
        lr_decay_step=args.lr_decay_step,
        lr_decay_step_2d=args.lr_decay_step_2d,
        lr_decay_rate=args.lr_decay_rate,
        bn_decay_step=args.bn_decay_step,
        bn_decay_rate=args.bn_decay_rate,
        loss_weights=loss_weights,
        use_vlm_align=args.use_vlm_align,
        scheduler_type=args.scheduler_type,
        begin_align_epoch=args.begin_align_epoch,
        save_pred=args.save_pred,
        ddp=args.ddp,
        reinit_epoch=args.reinit_epoch,
    )
    num_params = get_num_params(model)

    return solver, num_params, root, stamp

def save_info(args, root, num_params, train_dataset, val_dataset, exclude):
    info = {}
    for key, value in vars(args).items():
        if key not in exclude:
            info[key] = value
    
    info["num_train"] = len(train_dataset)
    info["num_val"] = len(val_dataset)
    info["num_train_scenes"] = len(train_dataset.scene_list)
    info["num_val_scenes"] = len(val_dataset.scene_list)
    info["num_params"] = num_params

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

    answer_vocab = train_dataset.answer_counter
    with open(os.path.join(root, "answer_vocab.json"), "w") as f:
        json.dump(answer_vocab, f, indent=4)        



def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])
    return scene_list

def get_scanqa(scanqa_train, scanqa_val, train_num_scenes, val_num_scenes):
    # get initial scene list
    train_scene_list = sorted(list(set([data["scene_id"] for data in scanqa_train])))
    val_scene_list = sorted(list(set([data["scene_id"] for data in scanqa_val])))

    # set train_num_scenes
    if train_num_scenes <= -1: 
        train_num_scenes = len(train_scene_list)
    else:
        assert len(train_scene_list) >= train_num_scenes

    # slice train_scene_list
    train_scene_list = train_scene_list[:train_num_scenes]

    # filter data in chosen scenes
    new_scanqa_train = []
    for data in scanqa_train:
        if data["scene_id"] in train_scene_list:
            new_scanqa_train.append(data)

    # set val_num_scenes
    if val_num_scenes <= -1: 
        val_num_scenes = len(val_scene_list)
    else:
        assert len(val_scene_list) >= val_num_scenes

    # slice val_scene_list
    val_scene_list = val_scene_list[:val_num_scenes]        

    new_scanqa_val = []
    for data in scanqa_val:
        if data["scene_id"] in val_scene_list:
            new_scanqa_val.append(data)

    #new_scanqa_val = scanqa_val[0:4] # debugging

    # all scanqa scene
    all_scene_list = train_scene_list + val_scene_list
    #print("train on {} samples and val on {} samples".format(len(new_scanqa_train), len(new_scanqa_val)))
    #exit()
    return new_scanqa_train, new_scanqa_val, all_scene_list


def train(args):
    # WandB init    
    from datetime import datetime  
    datetime = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    # sync datetime
    if args.ddp:
        datetimes = [None] * args.world_size
        torch.distributed.all_gather_object(datetimes, datetime)
        datetime = datetimes[0] # use the first one
    print("datetime: ", datetime)
    
    wandb.init(project="scanqa-new", config=args, group=datetime)

    # init training dataset
    print("preparing data...")
    scanqa_train, scanqa_val, all_scene_list = get_scanqa(SCANQA_TRAIN, SCANQA_VAL, args.train_num_scenes, args.val_num_scenes)
    scanqa = {
        "train": scanqa_train,
        "val": scanqa_val
    }

    # dataloader
    train_dataset, train_dataloader = get_dataloader(args, scanqa, all_scene_list, "train", DC, not args.no_augment, args.train_batch_size)
    val_dataset, val_dataloader = get_dataloader(args, scanqa, all_scene_list, "val", DC, False, args.val_batch_size)
    print("train on {} samples and val on {} samples".format(len(train_dataset), len(val_dataset)))

    dataloader = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    if args.use_vlm_align:
        scene_view_map = load_scene_view_map(args)

        print("encoding images with VLM...")
        if args.image_encoder_type == "blip":
            image_feat_dict, grid_size = compute_blip_view_features(all_scene_list, scene_view_map, topk=args.align_topk, alternative_ckpt=args.alternative_ckpt)
        elif args.image_encoder_type == "clip":
            image_feat_dict, grid_size = compute_clip_view_features(all_scene_list, scene_view_map, topk=args.align_topk, model_name=args.clip_model_name, ckpt_name=args.clip_ckpt_name, return_layer=args.clip_return_layer)
        setattr(args, 'image_feat_dict', image_feat_dict)
        setattr(args, 'grid_size', grid_size)
        setattr(args, 'scene_view_map', scene_view_map)
    else:
        setattr(args, 'image_feat_dict', None)
        setattr(args, 'grid_size', None)
        setattr(args, 'scene_view_map', None)

    print("initializing...")
    solver, num_params, root, stamp = get_solver(args, dataloader)
    solver.set_answer_vocab(train_dataset.answer_vocab)
    if stamp:
        wandb.run.name = stamp
        wandb.run.save()


    print("Start training...\n")
    save_info(args, root, num_params, train_dataset, val_dataset, exclude={"image_feat_dict", "scene_view_map"})
    solver(args.epoch, args.verbose)

if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    logger.setLevel(logging.INFO)
    
    log_format = (
        colorama.Fore.MAGENTA
        + "[%(asctime)s %(name)s %(levelname)s] "
        + colorama.Fore.WHITE
        + "%(message)s"
    )
    logging.basicConfig(
        format=log_format, level=logging.INFO, datefmt="%I:%M:%S"
    )

    args = parse_option()
    setattr(args, "slurm_job_id", os.environ.get("SLURM_JOB_ID", None))
    # args.slurm_gpus = os.environ.get("SLURM_GPUS", None)
    setattr(args, "slurm_gpus", os.environ.get("SLURM_GPUS", None))

    if args.ddp:
        args = init_distributed_mode(args)
    else:
        setattr(args, "world_size", 1)
        setattr(args, "local_rank", 0)

    if args.stage == "DET":
        args.no_answer = True
        args.use_aux_situation = False
        args.use_location_embedding = False
        args.sideload_obj_feature_path = ""
    elif args.stage == "VQA":
        ...

    if args.local_rank == 0:
        pprint(vars(args))
    # setting
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    

    train(args)
    
