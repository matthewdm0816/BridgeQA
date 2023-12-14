import os
import sys
import json
import pickle
import argparse
import importlib
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from datetime import datetime
from tqdm.auto import tqdm
from copy import deepcopy
from attrdict import AttrDict
from transformers import AutoTokenizer, AutoConfig
from argparse import Namespace

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder

from lib.config import CONF
from lib.dataset import ScannetQADataset
from lib.ap_helper import parse_predictions
from lib.loss_helper import get_loss
from models.qa_module import ScanQA
from utils.box_util import get_3d_box
# from utils.misc import overwrite_config
from data.scannet.model_util_scannet import ScannetDatasetConfig

project_name = "ScanQA_v1.0"
GLOBAL = Namespace()

def init_distributed_mode(args):
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    args.world_size = torch.distributed.get_world_size()
    args.local_rank = torch.distributed.get_rank()
    # args.device = torch.device("cuda", args.local_rank)
    torch.cuda.set_device(args.local_rank)
    return args

def get_dataloader(args, scanqa, all_scene_list, split, config):
    answer_vocab_path = os.path.join(CONF.PATH.OUTPUT, args.folder, "answer_vocab.json")
    answer_counter = json.load(open(answer_vocab_path))
    # answer_cands = sorted(answer_counter.keys())
    answer_cands = sorted(answer_counter.keys(), key=lambda x: (-answer_counter[x], x)) # sort by frequency
    config.num_answers = len(answer_cands)

    GLOBAL.answer_cands = answer_cands
    GLOBAL.answer_counter = answer_counter

    print("using {} answers".format(config.num_answers))

    if 'bert-' in args.tokenizer_name: 
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    else:
        tokenizer = None    

    # dataset = ScannetQADataset(
    #     scanqa=scanqa, 
    #     scanqa_all_scene=all_scene_list, 
    #     use_unanswerable=True,         
    #     answer_cands=answer_cands,
    #     answer_counter=answer_counter,
    #     answer_cls_loss=args.answer_cls_loss,
    #     split=split, 
    #     num_points=args.num_points, 
    #     use_height=(not args.no_height),
    #     use_color=args.use_color,         
    #     use_normal=args.use_normal, 
    #     use_multiview=args.use_multiview,
    #     tokenizer=tokenizer,
    # )
    dataset = ScannetQADataset(
        scanqa=scanqa, 
        scanqa_all_scene=all_scene_list, 
        use_unanswerable=True,  
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
        # augment=augment,
        # debug=args.debug,
        i2tfile=args.i2tfile if len(args.i2tfile_eval) == 0 else args.i2tfile_eval,
        scene_view_topk=args.align_topk,
        dset_views_path=args.dset_views_path,
    )
    print("predict for {} samples".format(len(dataset)))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    return dataset, dataloader


def get_model(args, config):
    # load tokenizer model
    if "bert-" in args.tokenizer_name:
        bert_model_name = args.tokenizer_name
        bert_config = AutoConfig.from_pretrained(bert_model_name)
        if hasattr(bert_config, "hidden_size"):
            lang_emb_size = bert_config.hidden_size
        else:
            # for distllbert
            lang_emb_size = bert_config.dim
    else:
        bert_model_name = None
        lang_emb_size = 300 # glove emb_size

    # load model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)

    # model = ScanQA(
    #     num_answers=config.num_answers,
    #     # proposal
    #     input_feature_dim=input_channels,            
    #     num_object_class=config.num_class, 
    #     num_heading_bin=config.num_heading_bin,
    #     num_size_cluster=config.num_size_cluster,
    #     mean_size_arr=config.mean_size_arr,
    #     num_proposal=args.num_proposals, 
    #     seed_feat_dim=args.seed_feat_dim,
    #     proposal_size=args.proposal_size,
    #     pointnet_width=args.pointnet_width,
    #     pointnet_depth=args.pointnet_depth,        
    #     vote_radius=args.vote_radius, 
    #     vote_nsample=args.vote_nsample,            
    #     # qa
    #     #answer_cls_loss="ce",
    #     answer_pdrop=args.answer_pdrop,
    #     mcan_num_layers=args.mcan_num_layers,
    #     mcan_num_heads=args.mcan_num_heads,
    #     mcan_pdrop=args.mcan_pdrop,
    #     mcan_flat_mlp_size=args.mcan_flat_mlp_size, 
    #     mcan_flat_glimpses=args.mcan_flat_glimpses,
    #     mcan_flat_out_size=args.mcan_flat_out_size,
    #     # lang
    #     lang_use_bidir=args.lang_use_bidir,
    #     lang_num_layers=args.lang_num_layers,
    #     lang_emb_size=lang_emb_size,
    #     lang_pdrop=args.lang_pdrop,
    #     bert_model_name=bert_model_name,
    #     freeze_bert=args.freeze_bert,
    #     finetune_bert_last_layer=args.finetune_bert_last_layer,
    #     # common
    #     hidden_size=args.hidden_size,
    #     # option
    #     use_object_mask=(not args.no_object_mask),
    #     use_lang_cls=(not args.no_lang_cls),
    #     use_reference=(not args.no_reference),
    #     use_answer=(not args.no_answer),            
    #     )
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
    )

    model_name = "model.pth"
    model_path = os.path.join(CONF.PATH.OUTPUT, args.folder, model_name)
    print('loading model from:', model_path)
    # to CUDA
    model = model.cuda()
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.")
    msg = model.load_state_dict(state_dict) #, strict=False)
    print(msg)
    model.eval()

    return model

def get_scanqa(args):
    scanqa = json.load(open(os.path.join(CONF.PATH.SCANQA, project_name + "_"+args.test_type+".json")))
    scene_list = sorted(list(set([data["scene_id"] for data in scanqa])))
    scanqa = [data for data in scanqa if data["scene_id"] in scene_list]
    return scanqa, scene_list

def predict(args):
    print("predict bounding boxes...")
    # constant
    DC = ScannetDatasetConfig()

    # init training dataset
    print("preparing data...")
    scanqa, scene_list = get_scanqa(args)

    # dataloader
    _, dataloader = get_dataloader(args, scanqa, scene_list, "test", DC)
    dataset = dataloader.dataset
    scanqa = dataset.scanqa


    setattr(args, 'image_feat_dict', None)
    setattr(args, 'grid_size', None)
    setattr(args, 'scene_view_map', None)

    # model
    model = get_model(args, DC)

    # config
    POST_DICT = {
        "remove_empty_box": True, 
        "use_3d_nms": True, 
        "nms_iou": 0.25,
        "use_old_type_nms": False, 
        "cls_nms": True, 
        "per_class_proposal": True,
        "conf_thresh": 0.05,
        "dataset_config": DC
    } if not args.no_nms else None

    if args.no_detection:
        POST_DICT = None

    # predict
    print("predicting...")
    pred_bboxes = []
    # for data_dict in tqdm(dataloader):
    #     # move to cuda
    #     for key in data_dict:
    #         if type(data_dict[key]) is dict:
    #             data_dict[key] = {k:v.cuda() for k, v in data_dict[key].items()}
    #         else:
    #             data_dict[key] = data_dict[key].cuda()
    dataloader = tqdm(dataloader)

    for i, data_dict in enumerate(dataloader):
        # move to cuda
        for key in data_dict:
            if type(data_dict[key]) is dict:
                data_dict[key] = {k:v.cuda() for k, v in data_dict[key].items()}
            elif type(data_dict[key]) is list or type(data_dict[key]) is str:
                # keep list as-is
                ...
            else:
                data_dict[key] = data_dict[key].cuda()   

        data_dict["phase"] = "val"
        data_dict["open_ended"] = args.open_ended      

        # feed
        with torch.no_grad():        
            data_dict = model(data_dict)
            if args.open_ended:
                for i in range(len(data_dict["open_ended_answer"])):
                    scanqa_idx = data_dict["scan_idx"][i].item()
                    pred_data = {
                        "scene_id": scanqa[scanqa_idx]["scene_id"],
                        "question_id": scanqa[scanqa_idx]["question_id"],
                        "open_ended_answer": data_dict["open_ended_answer"][i],
                    }
                    pred_bboxes.append(pred_data)  
                continue
            _, data_dict = get_loss(
                data_dict=data_dict, 
                config=DC, 
                detection=False,
                use_reference=not args.no_reference, 
                use_lang_classifier=not args.no_lang_cls,
                use_answer=(not args.no_answer),
            )
            

        objectness_preds_batch = torch.argmax(data_dict['objectness_scores'], 2).long()

        if POST_DICT:
            _ = parse_predictions(data_dict, POST_DICT)
            nms_masks = torch.LongTensor(data_dict['pred_mask']).cuda()
            # construct valid mask
            pred_masks = (nms_masks * objectness_preds_batch == 1).float()
        else:
            # construct valid mask
            pred_masks = (objectness_preds_batch == 1).float()

        # bbox prediction
        pred_ref = torch.argmax(data_dict['cluster_ref'] * pred_masks, 1) # (B,)
        pred_center = data_dict['center'] # (B,K,3)
        pred_heading_class = torch.argmax(data_dict['heading_scores'], -1) # B,num_proposal
        pred_heading_residual = torch.gather(data_dict['heading_residuals'], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
        pred_heading_class = pred_heading_class # B,num_proposal
        pred_heading_residual = pred_heading_residual.squeeze(2) # B,num_proposal
        pred_size_class = torch.argmax(data_dict['size_scores'], -1) # B,num_proposal
        pred_size_residual = torch.gather(data_dict['size_residuals'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
        pred_size_class = pred_size_class
        pred_size_residual = pred_size_residual.squeeze(2) # B,num_proposal,3

        topk = 10
        pred_answers_top10 = data_dict['answer_scores'].topk(topk, dim=1)[1]
        pred_answer_idxs = pred_answers_top10.tolist()

        for i in range(pred_ref.shape[0]):
            # compute the iou
            pred_ref_idx = pred_ref[i]
            pred_obb = DC.param2obb(
                pred_center[i, pred_ref_idx, 0:3].detach().cpu().numpy(), 
                pred_heading_class[i, pred_ref_idx].detach().cpu().numpy(), 
                pred_heading_residual[i, pred_ref_idx].detach().cpu().numpy(),
                pred_size_class[i, pred_ref_idx].detach().cpu().numpy(), 
                pred_size_residual[i, pred_ref_idx].detach().cpu().numpy()
            )
            pred_bbox = get_3d_box(pred_obb[3:6], pred_obb[6], pred_obb[0:3])

            # answer
            #pred_answer = dataset.answer_vocab.itos(pred_answer_idxs[i])
            pred_answers_top10 = [dataset.answer_vocab.itos(pred_answer_idx) for pred_answer_idx in pred_answer_idxs[i]]

            # store data
            scanqa_idx = data_dict["scan_idx"][i].item()
            pred_data = {
                "scene_id": scanqa[scanqa_idx]["scene_id"],
                "question_id": scanqa[scanqa_idx]["question_id"],
                "answer_top10": pred_answers_top10,
                "bbox": pred_bbox.tolist(),
                "2d_self_attention": data_dict["2d_self_attention"][i].detach().cpu().numpy().tolist(),
                "3d_self_attention": data_dict["3d_self_attention"][i].detach().cpu().numpy().tolist(),
                "2d_cross_attention": data_dict["2d_cross_attention"][i].detach().cpu().numpy().tolist(),
                "3d_cross_attention": data_dict["3d_cross_attention"][i].detach().cpu().numpy().tolist(),
            }
            pred_bboxes.append(pred_data)

    # dump
    print("dumping...")
    if args.open_ended:
        pred_path = os.path.join(CONF.PATH.OUTPUT, args.folder, "pred."+args.test_type+args.midfix+".open_ended.json")
    else:
        pred_path = os.path.join(CONF.PATH.OUTPUT, args.folder, "pred."+args.test_type+args.midfix+".json")

    with open(pred_path, "w") as f:
        json.dump(pred_bboxes, f, indent=4)

    print("done!")

def overwrite_config(args, past_args):
    for k, v in past_args.items():
        if hasattr(args, k) and getattr(args, k, None) is not None: # skip if args has past_args
            continue
        if k in ["i2tfile", "dset_views_path"] and getattr(args, k, None) is not None:
            continue
        setattr(args, k, v)
    return args   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="Folder containing the model")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--test_type", type=str, help="test_w_obj or test_wo_obj", default="test_wo_obj")
    parser.add_argument("--batch_size", type=int, help="batch size", default=8)
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--no_nms", action="store_true", help="do NOT use non-maximum suppression for post-processing.")
    parser.add_argument("--trial", type=int, default=-1, help="trial number")    
    parser.add_argument("--i2tfile", type=str, default=None, help="")
    parser.add_argument("--open_ended", action="store_true", help="use open-ended evaluation")
    parser.add_argument("--midfix", type=str, default="", help="midfix")
    # --i2tfile "/scratch/mowentao/BLIP/scene_eval_scanqa_interrogative_video.pkl" \
    args = parser.parse_args()
    train_args = json.load(open(os.path.join(CONF.PATH.OUTPUT, args.folder, "info.json")))
    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # overwrite
    args = init_distributed_mode(args)
    args = overwrite_config(args, train_args)
    seed = args.seed

    # reproducibility
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    predict(args)