import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")

DSET_PATH_SCANQA = {
    "test_w_obj": os.path.join(DATA_DIR, "qa/ScanQA_v1.0_test_w_obj.json"),
    "test_wo_obj": os.path.join(DATA_DIR, "qa/ScanQA_v1.0_test_wo_obj.json"),
    "train": os.path.join(DATA_DIR, "qa/ScanQA_v1.0_train.json"),
    "val": os.path.join(DATA_DIR, "qa/ScanQA_v1.0_val.json"),
}
DSET_PATH_SQA = {
    "test": os.path.join(DATA_DIR, "qa/SQA_test.json"),
    "train": os.path.join(DATA_DIR, "qa/SQA_train_scanqa.json"),
    "val": os.path.join(DATA_DIR, "qa/SQA_val.json"),
}


START_METHOD = "forkserver"
DSET_VIEWS_PATH = os.path.join(DATA_DIR, "scene_views_aligned")
SCAN_NAMES = list(
    filter(
        lambda n: n.endswith("00"),
        sorted(
            [
                line.rstrip()
                for line in open(
                    os.path.join(DATA_DIR, "scannet/meta_data/scannetv2.txt")
                )
            ]
        ),
    )
)
SCENE_FEAT_PATH = "scene_blip_features.pkl"

def is_dummy(answers):
    return len(answers) == 1 and answers[0] == ""


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method(START_METHOD, force=True)
    import torch
    import torch.multiprocessing
    from PIL import Image
    import validators
    import requests
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode

    torch.multiprocessing.set_start_method(START_METHOD, force=True)
    torch.multiprocessing.set_sharing_strategy("file_system")
    print(torch.multiprocessing.get_start_method())
    from torch.nn.parallel import DataParallel, DistributedDataParallel
    import pretty_errors
    import os, sys, json, glob, logging, random, pickle, warnings, colorama, datasets, toml, transformers
    from icecream import ic
    from collections import defaultdict, Counter
    from datetime import datetime
    from pprint import pprint
    from datasets import (
        Dataset,
        DatasetDict,
        Features,
        Sequence,
        Value,
        load_dataset,
        load_metric,
    )
    from torch.utils.data import DataLoader
    import torch.nn.functional as F
    from tqdm.auto import tqdm
    from transformers import AutoTokenizer
    import itertools
    import numpy as np
    from typing import Dict, List, Any, Optional, Union, Set

    sys.path.append(".")
    from multiprocessing.spawn import freeze_support
    from utils_eval_blip import *
    from copy import copy, deepcopy

    freeze_support()
    seed_all(42)

    local_rank = 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    logger = init_logger_nonddp()

    def parse_args():
        from argparse import ArgumentParser

        parser = ArgumentParser()
        # --- OPTIONS BEGIN ---
        parser.add_argument("--topk", type=int, default=3) # useless
        parser.add_argument("--outfile", type=str, default="scene_eval.json")
        parser.add_argument("--split", type=str, default="all")
        parser.add_argument("--dryrun", action="store_true")
        parser.add_argument("--topk_images", type=int, default=1)
        parser.add_argument("--random", action="store_true")
        parser.add_argument("--use_composed_qa", action="store_true")
        parser.add_argument("--composed_qa_json", type=str, default="/home/mowentao/scratch/toys/composed_decl_from_qa.json")
        parser.add_argument("--dataset", type=str, default="scanqa")
        parser.add_argument("--dset_views_path", type=str, default=DSET_VIEWS_PATH)
        parser.add_argument("--nocheck_blank", action="store_true")
        # --- OPTIONS END ---
        return parser.parse_args()

    args = parse_args()

    if args.dataset == "scanqa":
        DSET_PATH = DSET_PATH_SCANQA
    elif args.dataset == "sqa":
        DSET_PATH = DSET_PATH_SQA

    if args.split == "all":
        args.split = list(DSET_PATH.keys())
    else:
        args.split = args.split.strip().split(",")

    print(args.split)

    logger.info(f"--- OPTIONS BEGIN ---")
    for k, v in args.__dict__.items():
        logger.info(f"{k}: {v}")
    logger.info(f"--- OPTIONS END ---")

    world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

    if args.use_composed_qa:
        composed_qa = json.load(open(args.composed_qa_json, "r"))

    # --- Load scanqa dset
    def load_dset(dset_path):
        import json

        logging.info("Loading Dataset...")
        dset_dict = {}
        for split, p in dset_path.items():
            data = json.load(open(p, "r"))
            dset = datasets.Dataset.from_dict(
                {k: [(s[k] if k != "question_id" else str(s[k])) if k in s else None for s in data] for k in data[0].keys()}
            )
            dset_dict[split] = dset
        return datasets.DatasetDict(dset_dict)

    dset = load_dset(DSET_PATH)

    # add answers = [] for empty answers
    for split in args.split:
        if "answers" not in dset[split].features:
            dset[split] = dset[split].map(lambda x: {"answers": [[""] for _ in range(len(x["question"]))]}, batched=True)
    # remove object info if exists
    for split in args.split:
        dset[split] = dset[split].remove_columns(list({"object_ids", "object_names"}.intersection(dset[split].column_names)))

    logger.info(dset)

    all_scans = []

    def cnt_scans(s):
        global all_scans
        all_scans += s["scene_id"]


    dset = datasets.concatenate_datasets([dset[split] for split in args.split])
    dset.map(cnt_scans, batched=True)
    SCAN_NAMES = sorted(set(all_scans).intersection(SCAN_NAMES))

    logger.info(f"Total {len(SCAN_NAMES)} scenes.")


    pool = SceneViewsPool(args.dset_views_path, SCAN_NAMES, preprocess=preprocess, nocheck_blank=args.nocheck_blank)
    images = pool.images

    # --- Init BLIP Model
    # from models.blip import blip_decoder
    # from models.blip_pretrain import blip_pretrain
    # from models.blip_vqa import blip_vqa
    from models.blip_itm import blip_itm

    logger.info("Loading BLIP Models...")

    image_size = 384
    # model_url = "ckpts/model_large.pth"
    model_url = "ckpts/model_large_retrieval_coco.pth"

    model = blip_itm(pretrained=model_url, image_size=image_size, vit="large")


    if world_size > 1:
        # Use multi-gpu
        model.visual_encoder = DataParallel(
            model.visual_encoder,
            device_ids=list(range(world_size)),
            output_device=device,
        )
        model.text_encoder = DataParallel(
            model.text_encoder, device_ids=list(range(world_size)), output_device=device
        )

    model.eval()
    model = model.to(device)

    # if not feature_exists:
    # --- Encode BLIP image features
    def encode_feature(model_image, images):
        logging.info("Beginning Encoding Images...")
        image_feat_dict = {}
        for scan_name, img_dict in tqdm(images.items()):
            dataloader = DataLoader(list(img_dict.items()), batch_size=256)
            image_feat_dict[scan_name] = {}
            with torch.no_grad():
                for batch in dataloader:
                    img_names, images = batch
                    images = images.to(device)
                    image_embeds: torch.Tensor = model_image(images).to(device)
                    for i, img_name in enumerate(img_names):
                        image_feat_dict[scan_name][img_name] = image_embeds[i].cpu()
        return image_feat_dict

    image_feat_dict = encode_feature(model.visual_encoder, images)

    # --- Prediction
    pred = {}
    pred_answer = {}
    itm_scores = {}
    for scan_name in tqdm(SCAN_NAMES):
        dset_scan = dset.filter(lambda s: s["scene_id"] == scan_name)
        logging.info(f"{dset_scan.num_rows} questions for {scan_name}")
        if dset_scan.num_rows == 0:
            continue

        scan_name_old = deepcopy(scan_name)
        if not scan_name.endswith("00"):
            scan_name[-2:] = "00"

        image_names = sorted(image_feat_dict[scan_name].keys())
        total_images = len(image_names)
        logging.info(f"{total_images} images for {scan_name_old}")
        if total_images == 0:
            continue

        # --- Begin simiarity evaluation
        with torch.no_grad():
            image_feats = torch.stack(
                [image_feat_dict[scan_name][n] for n in image_names]
            ).to(device)
            # image_feats = model.vision_proj(image_feats[:, 0, :])
            image_feats = F.normalize(model.vision_proj(image_feats[:, 0, :]), dim=-1)

            scan_images = torch.stack([images[scan_name][n] for n in image_names]).to(
                device
            )

            # BLIP ITM evaluation
            questions = dset_scan["question"]
            question_ids = dset_scan["question_id"]
            # stringify
            question_ids = [str(qid) for qid in question_ids]

            bs = len(questions)

            if args.use_composed_qa:
                questions_text = [composed_qa[qid] for qid in question_ids]
            else:
                questions_text = questions

            text = model.tokenizer(
                questions_text,
                padding="max_length",
                truncation=True,
                max_length=70,
                return_tensors="pt",
            ).to(image_feats.device)

            text_output = model.text_encoder(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode="text",
            )
            # text_feat = model.text_proj(text_output.last_hidden_state[:, 0, :])
            text_feat = F.normalize(
                model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
            )

            sim = text_feat @ image_feats.t()  # [B, N_img]
            topk_pred = sim.topk(k=total_images).indices.cpu().tolist()  # [B, K]
            ic(sim.shape)

            # record
            for i, question_id in enumerate(question_ids):
                pred[question_id] = [image_names[iid] for iid in topk_pred[i]]
                itm_scores[question_id] = [sim[i, iid] for iid in topk_pred[i]]


    # --- Save prediction
    if not args.dryrun:
        # json.dump({"view": pred, "answer": pred_answer, "itm_scores": itm_scores}, open(args.outfile, "w"))
        torch.save({"view": pred, "answer": pred_answer, "itm_scores": itm_scores}, args.outfile)

    logging.info("Finished Prediction")
