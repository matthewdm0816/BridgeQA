import torch
from typing import List, Union, Optional, Any, Set
import logging
from icecream import ic
import colorama
import numpy as np
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import glob
import os, sys

logger = logging.getLogger(__name__)


def init_logger_nonddp():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_format = (
        colorama.Fore.MAGENTA
        + "[%(asctime)s %(name)s %(levelname)s] "
        + colorama.Fore.WHITE
        + "%(message)s"
    )
    logging.basicConfig(format=log_format, level=logging.INFO, datefmt="%I:%M:%S")

    return logger


def seed_all(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def recursive_stack(
    maybe_tensors: Union[List[List], List[torch.Tensor]]
) -> torch.Tensor:
    if isinstance(maybe_tensors[0], torch.Tensor):
        return torch.stack(maybe_tensors)
    elif not isinstance(maybe_tensors[0], list):
        return maybe_tensors  # maybe not tensors at all, like strings or ids
    else:
        return torch.stack(
            [recursive_stack(sub_maybe_tensors) for sub_maybe_tensors in maybe_tensors]
        )


def collate_features(batch):
    bs = len(batch)
    result = dict()
    # ic(batch[0].keys())
    for key in batch[0].keys():

        if isinstance(batch[0][key], str):
            # simply put into a list of strings
            result[key] = [batch[idx][key] for idx in range(bs)]
        else:
            result[key] = recursive_stack([batch[idx][key] for idx in range(bs)])
    return result


def collate_features_simple(batch):
    bs = len(batch)
    result = dict()
    for key in batch[0].keys():
        # simply put into a list
        result[key] = [batch[idx][key] for idx in range(bs)]
    return result


class ImageListPool:
    @classmethod
    def _getid(cls, image_path):
        r"""
        "VizWiz-werwerwer-0001123.jpg" => 1123
        "COCO_train2014_00000000123123.jpg" => 123123
        """
        filename = os.path.basename(image_path)
        image_id = filename.split("_")[-1]
        image_id, _ = image_id.split(".")
        return int(image_id)

    def __init__(self, path_list, preprocess, init: bool = True):
        self.image_dict = dict()
        self.image_feat_dict = dict()
        self.path_list = [os.path.expanduser(path) for path in path_list]
        self.preprocess = preprocess
        self.use_h5 = False
        filelist = []
        for path in self.path_list:
            filelist += glob.glob(path)
        filelist = sorted(filelist)
        # filelist = filelist[:10000]
        self.filelist = filelist
        self.total_files = len(self.filelist)
        ic(self.total_files)
        if init:
            self.init()

    def __len__(self):
        return self.total_files

    def __getitem__(self, k):
        filename = self.filelist[k]
        iid = self._getid(filename)
        return iid, self.preprocess(Image.open(filename).convert("RGB"))

    def init(self, num_workers: int = 32):
        if num_workers < 1:
            for filename in tqdm(glob.glob(self.path)):
                image_id = self._getid(filename)
                image = self.preprocess(Image.open(filename))
                self.image_dict[image_id] = image
        else:
            from concurrent.futures import (
                ThreadPoolExecutor,
                wait,
                ProcessPoolExecutor,
            )
            from multiprocessing import Queue

            # q = Queue()
            executor = ThreadPoolExecutor(max_workers=num_workers)
            futures = []

            pbar = tqdm(total=self.total_files)

            for filename in self.filelist:
                future = executor.submit(self._load_single_image_mt, filename)
                future.add_done_callback(lambda future: pbar.update(1))
                futures.append(future)

            wait(futures)

    def _load_single_image_mt(self, filename):
        iid = self._getid(filename)
        image = self.preprocess(Image.open(filename).convert("RGB"))
        self.image_dict[iid] = image

    def encode(self, model_image, device):
        logging.info("Beginning Encoding Images...")
        dataloader = DataLoader(list(self.image_dict.items()), batch_size=32)
        with torch.no_grad():
            # for idx, img in tqdm(self.image_dict.items(), total=len(self.image_dict)):
            #     self[idx] = model.encode_image(img.unsqueeze(0).to(device)).squeeze(0)
            for batch in tqdm(dataloader, total=len(dataloader)):
                # ic(batch)
                iids, images = batch
                images = images.to(device)
                image_embeds: torch.Tensor = model_image(images).to(device)
                for i, iid in enumerate(iids):
                    self.image_feat_dict[iid.item()] = image_embeds[i].cpu()

    def load_from_h5(self, dset):
        self.use_h5 = True
        self.dset = dset

    def _load_feature_from_dset(self, k, dset):
        self.image_feat_dict[int(k)] = torch.tensor(dset[k][:])


from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


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


# Quesition to LM prompt
def preprocess_q(q: str) -> List[str]:
    return q.lower().replace("?", "").strip().split(" ")


def index_of_set(li: List[Any], set_range: Set[Any]) -> int:
    for idx, item in enumerate(li):
        if item in set_range:
            return idx
    raise ValueError


QVERBS = ["is", "are", "were", "was", "do", "does", "did"]
BEVERBS = ["is", "are", "were", "was"]

from functools import wraps


def default_return(default_f):
    def decorator(f):
        @wraps(f)
        def g(*args, **kwargs):
            result = f(*args, **kwargs)
            return result if result is not None else default_f(*args, **kwargs)

        return g

    return decorator


@default_return(default_f=lambda q: preprocess_q(q))
def question_to_prompt(question: str) -> List[str]:
    tokens = preprocess_q(question)
    question_type = tokens[0]
    if question_type == "what":
        try:
            first_verb_idx = index_of_set(tokens, QVERBS)
        except ValueError as e:
            return None
        interrogative_words = tokens[:first_verb_idx]
        if len(interrogative_words) == 1:
            # single "what"
            if tokens[first_verb_idx] in BEVERBS:
                return tokens[first_verb_idx + 1 :] + [tokens[first_verb_idx]]
            else:
                return tokens[first_verb_idx + 1 :]
        else:
            # "what color" like
            return (
                ["the", *tokens[1:first_verb_idx], "of"]
                + tokens[first_verb_idx + 1 :]
                + [tokens[first_verb_idx]]
            )
    elif question_type == "why":
        try:
            first_verb_idx = index_of_set(tokens, QVERBS)
        except ValueError as e:
            return None
        interrogative_words = tokens[:first_verb_idx]
        return ["the", "reason", "of"] + tokens[first_verb_idx + 1 :] + ["is"]
    elif question_type == "who":
        try:
            first_verb_idx = index_of_set(tokens, QVERBS)
        except ValueError as e:
            return None
        interrogative_words = tokens[:first_verb_idx]
        return ["the", "people"] + tokens[first_verb_idx + 1 :] + ["is"]
    elif question_type == "how":
        if tokens[1] == "many":
            try:
                first_verb_idx = index_of_set(tokens, QVERBS)
            except ValueError as e:
                return None
            return ["the", "number", "of"] + tokens[first_verb_idx + 1 :] + ["is"]
        else:
            # FIXME: NOT IMPLED
            return None
    else:
        # FIXME: NOT IMPLED
        return None


import subprocess as sp
import os


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = (
        sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    )
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


def preprocess_blip(raw_image):
    image_size = 384
    w, h = raw_image.size

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
    image = transform(raw_image)
    return image


def preprocess_blip_vqa(raw_image):
    image_size = 480
    w, h = raw_image.size

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
    image = transform(raw_image)
    return image


class SceneViewsPool:
    def __init__(self, DSET_VIEWS_PATH, SCAN_NAMES, preprocess, init: bool = True, eff_images=None, nocheck_blank: bool = False):
        self.images = dict()
        self.preprocess = preprocess
        self.SCAN_NAMES = SCAN_NAMES
        self.DSET_VIEWS_PATH = DSET_VIEWS_PATH
        self.nocheck_blank = nocheck_blank
        print(f"Loading all scene views from {DSET_VIEWS_PATH}...")
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
                p = os.path.join(self.DSET_VIEWS_PATH, scan_name)
                filelist = glob.glob(f"{p}/*.jpg")
                if len(filelist) == 0:
                    filelist = glob.glob(f"{p}/color/*.jpg")
                if len(filelist) == 0:
                    print(f"Warning: no images found in {p}!")

                if eff_images is not None:
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
        if self.nocheck_blank or not isblank(img, 0.7):
            self.images[scan_name][img_name] = self.preprocess(img)


from torch.utils.data.distributed import DistributedSampler


def get_ddp_dataloader(dataset, batch_size, shuffle, **kwargs) -> torch.utils.data.DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=DistributedSampler(dataset, seed=42, shuffle=shuffle),
        **kwargs,
    )

def get_scan_dataloaders(dset_scan, dset_scan_val, world_size, args):
    if world_size == 1:
        dataloader_train = DataLoader(
            dset_scan,
            batch_size=args.train_batch_size // args.topk_images,
            collate_fn=collate_features_simple,
            shuffle=False,
        )
        dataloader_val = DataLoader(
            dset_scan_val,
            batch_size=args.eval_batch_size // args.topk_images,
            collate_fn=collate_features_simple,
        )
    else:
        dataloader_train = get_ddp_dataloader(
            dset_scan,
            batch_size=args.train_batch_size // args.topk_images,
            shuffle=False,
            collate_fn=collate_features_simple,
        )
        dataloader_val = get_ddp_dataloader(
            dset_scan_val,
            batch_size=args.eval_batch_size // args.topk_images,
            shuffle=False,
            collate_fn=collate_features_simple,
        )
    return dataloader_train, dataloader_val

import torch.distributed as dist


def all_gather_concat_list(lst, world_size):
    lists = [None] * world_size
    dist.all_gather_object(lists, lst)
    return sum(lists, [])

class Averager:
    def __init__(self) -> None:
        self.n, self.v = 0, 0.0

    def add(self, x: float, n: int = 1) -> None:
        self.v += x
        self.n += n
    
    def value(self):
        return self.v / self.n if self.n > 0 else 0.0


