import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "..", "data")

DSET_PATH = {
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
SCENE_FEAT_PATH = os.path.join(DATA_DIR, "scene_blip_features.pkl")
BLIP_PATH = CURRENT_DIR

import multiprocessing
from multiprocessing.spawn import freeze_support

freeze_support()
import torch
import torch.multiprocessing
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import pretty_errors
import os, sys, json, glob, logging, random, pickle, warnings, colorama, datasets, toml, transformers, re
from icecream import ic
from collections import defaultdict, Counter
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

import torch.distributed as dist

from models.vit import VisionTransformer
from models.blip_vqa_3d import blip_vqa3d
import math
from time import time


sys.path.append(".")

logger = logging.getLogger(__name__)

def preprocess(image):
    image_size = 384
    transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    image = transform(image)
    return image


def preprocess_vqa(image):
    image_size = 480
    transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    image = transform(image)
    return image

def preprocess_caption(image):
    image_size = 384
    transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    image = transform(image)
    return image

def isblank(image, ratio=0.5):
    pix = torch.from_numpy(np.asarray(image))  # [HWC]
    blank = (pix == 255).all(dim=-1).sum().item()
    total_pixels = pix.size(0) * pix.size(1)
    return blank > total_pixels * ratio

class SceneViewsPool:
    def __init__(self, dset_views_path, SCAN_NAMES, preprocess, init: bool = True, eff_images=None, check_blank: bool = False):
        self.images = dict()
        self.poses = dict()
        self.depths = dict()
        self.preprocess = preprocess
        self.SCAN_NAMES = SCAN_NAMES
        self.DSET_VIEWS_PATH = dset_views_path
        self.check_blank = check_blank
        if init:
            self.init(eff_images=eff_images)

    def init(self, num_workers: int = 32, eff_images=None):
        if eff_images is None:
            print("Loading all scene views...")
        if num_workers < 1:
            # Deprecated
            for filename in tqdm(glob.glob(self.path)):
                image_id = self._getid(filename)
                image = self.preprocess(Image.open(filename))
                self.image_dict[image_id] = image
        else:
            from concurrent.futures import (
                ThreadPoolExecutor,
                wait,
            )

            executor = ThreadPoolExecutor(max_workers=num_workers)
            futures = []
            
            total_files = 0
            for scan_name in tqdm(self.SCAN_NAMES):
                self.images[scan_name] = {}
                self.poses[scan_name] = {}
                self.depths[scan_name] = {}
                p = os.path.join(self.DSET_VIEWS_PATH, scan_name)
                filelist = glob.glob(f"{p}/*.jpg")
                if len(filelist) == 0:
                    filelist = glob.glob(f"{p}/color/*.jpg")
                if len(filelist) == 0:
                    print(f"Warning: no images found in {p}!")

                if eff_images is not None:
                    if scan_name not in eff_images:
                        filelist = []
                    else:
                        eff_inames = eff_images[scan_name]
                        filelist = list(filter(lambda fname: os.path.basename(fname) in eff_inames, filelist))
                
                total_files += len(filelist)
            print(f"loading {total_files} scene views...")

            pbar = tqdm(total=total_files, miniters=1_000, mininterval=float("inf"))

            for scan_name in self.SCAN_NAMES:
                p = os.path.join(self.DSET_VIEWS_PATH, scan_name)
                filelist = glob.glob(f"{p}/*.jpg")
                if len(filelist) == 0:
                    filelist = glob.glob(f"{p}/color/*.jpg")
                if len(filelist) == 0:
                    print(f"Warning: no images found in {p}!")

                if eff_images is not None:
                    if scan_name not in eff_images:
                        filelist = []
                    else:
                        eff_inames = eff_images[scan_name]
                        filelist = list(filter(lambda fname: os.path.basename(fname) in eff_inames, filelist))
                
                for filename in filelist:
                    future = executor.submit(
                        self._load_single_image_mt, scan_name, filename
                    )
                    future.add_done_callback(lambda future: pbar.update(1))
                    futures.append(future)

            wait(futures)

    def _load_single_image_mt(self, scan_name, filename):
        img_name = os.path.basename(filename)
        img = Image.open(filename).convert("RGB")
        if "/color/" in filename:
            # load pose from e.g., /scratch/generalvision/ScanQA-feature/frames_square/scene0000_00/pose/xxx.txt
            pose_path = filename.replace("/color/", "/pose/").replace(".jpg", ".txt")
            pose = np.loadtxt(pose_path, dtype=np.float32) # [4, 4]
            self.poses[scan_name][img_name] = pose
            # load depths from e.g., /scratch/generalvision/ScanQA-feature/frames_square/scene0000_00/depth/xxx.png
            depth_path = filename.replace("/color/", "/depth/").replace(".jpg", ".png")
            depth = np.asarray(Image.open(depth_path)) # [H, W], typically 41*32
            depth = depth.astype(np.float32) / 1000.0
            self.depths[scan_name][img_name] = depth
        else:
            self.poses[scan_name][img_name] = None
            self.depths[scan_name][img_name] = None
        if not (self.check_blank and isblank(img, 0.7)):
            self.images[scan_name][img_name] = self.preprocess(img)

def load_scene_view_map(i2tfile):
    logger.info("Loading scene view map...")
    if i2tfile.endswith(".pkl"):
        tmp = torch.load(i2tfile, map_location="cpu")
    else:
        tmp = json.load(open(i2tfile, "r"))
    pred = tmp["view"]
    return pred


def get_blip_model(i2tfile, i2tfile_eval, topk=1, alternative_ckpt: str="", dset_views_path=None, dset="sqa", **kwargs):
    device = torch.device("cuda", dist.get_rank()) 

    # calculate qid->scene id
    # load annotations
    data = []
    if dset == "sqa":
        dset_path = DSET_PATH_SQA
    elif dset == "scanqa":
        dset_path = DSET_PATH
    else:
        raise NotImplementedError
    
    for path in dset_path.values():
        data.extend(json.load(open(path)))
    qid2scene = {d["question_id"]: d["scene_id"] for d in data}

    scene_view_map = load_scene_view_map(i2tfile)
    assert scene_view_map is not None, "Provide q-view mapping to load less images"  
    eff_images: dict[str, list] = {}  
    for qid, pred in scene_view_map.items():
        # scene_id = f"{qid.strip().split('-')[1]}_00"
        try:
            scene_id = qid2scene[int(qid)]
        except:
            scene_id = qid2scene[str(qid)]

        image_names = pred[:topk]
        if scene_id not in eff_images:
            eff_images[scene_id] = []
        eff_images[scene_id].extend(image_names)

    # --- Load views
    # if scene_ids is None:
    scene_ids = SCAN_NAMES
    if dset_views_path is None:
        dset_views_path = DSET_VIEWS_PATH
    pool = SceneViewsPool(dset_views_path, scene_ids, preprocess=preprocess_vqa, eff_images=eff_images, check_blank=(dset_views_path==DSET_VIEWS_PATH))
    images = pool.images
    poses = pool.poses
    depths = pool.depths

    if i2tfile_eval != i2tfile:
        scene_view_map_eval = load_scene_view_map(i2tfile_eval)
        eff_images_eval: dict[str, list] = {}  
        for qid, pred in scene_view_map_eval.items():
            # scene_id = f"{qid.strip().split('-')[1]}_00"
            try:
                scene_id = qid2scene[int(qid)]
            except:
                scene_id = qid2scene[str(qid)]

            image_names = pred[:topk]
            if scene_id not in eff_images_eval:
                eff_images_eval[scene_id] = []
            eff_images_eval[scene_id].extend(image_names)
        pool_eval = SceneViewsPool(dset_views_path, scene_ids, preprocess=preprocess_vqa, eff_images=eff_images_eval, check_blank=(dset_views_path==DSET_VIEWS_PATH))
        images_eval = pool_eval.images
        poses_eval = pool_eval.poses
        depths_eval = pool_eval.depths
    else:
        images_eval = images
        scene_view_map_eval = scene_view_map
        poses_eval = poses
        depths_eval = depths
    

    # --- Init BLIP Model
    logger.info("Loading BLIP Models...")

    model = blip_vqa3d(
        pretrained=os.path.join(BLIP_PATH, "ckpts/model_base_vqa_capfilt_large.pth"), 
        image_size=480, 
        vit="base",
        **kwargs 
        # num_answers=num_answers, 
        # scene_size=scene_size,
    )
    if alternative_ckpt != "":
        print(f"Loading alternative BLIP model from {alternative_ckpt}")
        model.load_state_dict(torch.load(alternative_ckpt))
    model.train()
    model = model.to(device)
    grid_size = model.visual_encoder.patch_embed.grid_size
    logger.info(f"BLIP grid size: {grid_size}")

    return (images, images_eval), model, (scene_view_map, scene_view_map_eval), (poses, poses_eval), (depths, depths_eval), grid_size

def get_blip_model_simple(alternative_ckpt: str="", random_init_blip=False, **kwargs):
    # --- Init BLIP Model
    device = torch.device("cuda", dist.get_rank()) 
    logger.info("Loading BLIP Models...")
    # pretrained = os.path.join(BLIP_PATH, "ckpts/model_base_vqa_capfilt_large.pth") if not random_init_blip else None
    model = blip_vqa3d(
        pretrained=os.path.join(BLIP_PATH, "../ckpts/model_base_vqa_capfilt_large.pth"),
        image_size=480, 
        vit="base",
        random_init_blip=random_init_blip,
        **kwargs
        # num_answers=num_answers, 
        # scene_size=scene_size,
    )
    if alternative_ckpt != "":
        print(f"Loading alternative BLIP model from {alternative_ckpt}")
        model.load_state_dict(torch.load(alternative_ckpt))
    model.train()
    model = model.to(device)
    grid_size = model.visual_encoder.patch_embed.grid_size
    logger.info(f"BLIP grid size: {grid_size}")

    return model, grid_size


def repeat_elements(lst, n):
    # Create a new list to store the repeated elements
    new_lst = []
    
    # Iterate over each element in the input list
    for element in lst:
        # Append the element to the new list n times
        new_lst.extend([element] * n)
    
    return new_lst


def all_gather_concat_list(lst, world_size):
    lists = [None] * world_size
    dist.all_gather_object(lists, lst)
    return sum(lists, [])




