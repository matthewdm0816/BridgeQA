DSET_PATH = {
    "test_w_obj": "/home/mowentao/data/ScanQA/data/qa/ScanQA_v1.0_test_w_obj.json",
    "test_wo_obj": "/home/mowentao/data/ScanQA/data/qa/ScanQA_v1.0_test_wo_obj.json",
    "train": "/home/mowentao/data/ScanQA/data/qa/ScanQA_v1.0_train.json",
    "val": "/home/mowentao/data/ScanQA/data/qa/ScanQA_v1.0_val.json",
}
START_METHOD = "forkserver"
DSET_VIEWS_PATH = "/home/mowentao/data/ScanQA/data/scene_views_aligned"
SCAN_NAMES = list(
    filter(
        lambda n: n.endswith("00"),
        sorted(
            [
                line.rstrip()
                for line in open(
                    "/home/mowentao/data/ScanQA/data/scannet/meta_data/scannetv2.txt"
                )
            ]
        ),
    )
)
SCENE_FEAT_PATH = "/home/mowentao/data/ScanQA/data/scene_blip_features.pkl"
BLIP_PATH = "/home/mowentao/scratch/BLIP"

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


from models.vit import VisionTransformer
from models.blip_vqa import blip_vqa
import math
from time import time

sys.path.append(".")
from BLIP.utils_eval_blip import *

logger = logging.getLogger(__name__)

def load_scene_view_map(args):
    tmp = json.load(open(args.i2tfile, "r"))
    pred = tmp["view"]
    return pred


def compute_blip_view_features(scene_ids=None, scene_view_map=None, topk=1, alternative_ckpt: str=""):
    r"""
    Get all scene views + BLIP features
    """
    print(f"Loading top-{topk} images")
    seed_all(42)

    local_rank = 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    assert scene_view_map is not None, "Provide q-view mapping to load less images"  
    eff_images: dict[str, list] = {}  
    for qid, pred in scene_view_map.items():
        scene_id = f"{qid.strip().split('-')[1]}_00"
        image_names = pred[:topk]
        if scene_id not in eff_images:
            eff_images[scene_id] = []
        eff_images[scene_id].extend(image_names)

    # --- Load views
    if scene_ids is None:
        scene_ids = SCAN_NAMES
    pool = SceneViewsPool(DSET_VIEWS_PATH, scene_ids, preprocess=preprocess_vqa, eff_images=eff_images)
    images = pool.images

    # --- Init BLIP Model
    logger.info("Loading BLIP Models...")

    model = blip_vqa(
        pretrained=os.path.join(BLIP_PATH, "ckpts/model_base_vqa_capfilt_large.pth"), image_size=480, vit="base"
    )
    if alternative_ckpt != "":
        print(f"Loading alternative BLIP model from {alternative_ckpt}")
        model.load_state_dict(torch.load(alternative_ckpt))
    model.eval()
    model = model.to(device)
    grid_size = model.visual_encoder.patch_embed.grid_size
    logger.info(f"BLIP grid size: {grid_size}")

    # --- Encode BLIP image features
    @torch.no_grad()
    def encode_feature(model_image, images):
        feature_shape = None
        logging.info("Beginning Encoding Images...")
        image_feat_dict = {}
        for scan_name, img_dict in tqdm(images.items()):
            dataloader = DataLoader(list(img_dict.items()), batch_size=256)
            image_feat_dict[scan_name] = {}
            with torch.no_grad():
                for batch in dataloader:
                    img_names, images = batch
                    images = images.to(device)
                    image_embeds: torch.Tensor = model_image(images, return_fm=-2).to(device) # [B, N, C]
                    G = int(image_embeds.size(1) ** 0.5)
                    image_embeds = image_embeds[:, 1:].view(-1, G, G, image_embeds.size(-1)) # remove [CLS], [B, G, G, C]
                    # record cumsum only, for fast mean pooling
                    image_embeds = image_embeds.cumsum(dim=1)
                    image_embeds = image_embeds.cumsum(dim=2)
                    image_embeds = F.pad(image_embeds, (0, 0, 1, 0, 1, 0), "constant", 0) # [B, 1+G, 1+G, C]
                    feature_shape = image_embeds.shape[1:]
                    # feature_shape = image_embeds.shape
                    for i, img_name in enumerate(img_names):
                        image_feat_dict[scan_name][img_name] = image_embeds[i].cpu()
        return image_feat_dict, feature_shape

    image_feat_dict, feature_shape = encode_feature(model.visual_encoder, images)

    # add all dismissed images' feature to be dummy nan # why?
    # for scan_name, img_list in eff_images.items():
    #     if scan_name in scene_ids:
    #         for img_name in img_list:
    #             if img_name not in image_feat_dict[scan_name]:
    #                 print(scan_name, img_name)
    #                 image_feat_dict[scan_name][img_name] = torch.zeros(feature_shape) # + torch.nan

    logging.info("Finished Pre-Computation")
    return image_feat_dict, grid_size


def compute_clip_view_features(scene_ids=None, scene_view_map=None, topk=1, alternative_ckpt: str="", model_name="ViT-B-16", ckpt_name='laion2b_s34b_b88k', return_layer=-1):
    r"""
    Get all scene views + CLIP features
    """
    print(f"Loading top-{topk} images")
    seed_all(42)

    local_rank = 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # --- Init CLIP Model
    logger.info(f"Loading CLIP Model {model_name}-{ckpt_name}...")
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=ckpt_name)
    model.eval()
    model = model.to(device)
    grid_size = model.visual.grid_size
    logger.info(f"CLIP grid size: {grid_size}")

    

    assert scene_view_map is not None, "Provide q-view mapping to load less images"  
    eff_images: dict[str, list] = {}  
    for qid, pred in scene_view_map.items():
        scene_id = f"{qid.strip().split('-')[1]}_00"
        image_names = pred[:topk]
        if scene_id not in eff_images:
            eff_images[scene_id] = []
        eff_images[scene_id].extend(image_names)

    # --- Load views
    if scene_ids is None:
        scene_ids = SCAN_NAMES
    pool = SceneViewsPool(DSET_VIEWS_PATH, scene_ids, preprocess=preprocess, eff_images=eff_images)
    images = pool.images

    # --- Encode CLIP image features
    @torch.no_grad()
    def encode_feature(clip_model: open_clip.CLIP, images):
        feature_shape = None
        logging.info("Beginning Encoding Images...")
        image_feat_dict = {}
        for scan_name, img_dict in tqdm(images.items()):
            dataloader = DataLoader(list(img_dict.items()), batch_size=768)
            image_feat_dict[scan_name] = {}
            with torch.no_grad():
                for batch in dataloader:
                    img_names, images = batch
                    images = images.to(device)
                    image_embeds: torch.Tensor = clip_model.encode_image_gridfeature(images, return_layer=return_layer).to(device) # [B, N, C]
                    G = int(image_embeds.size(1) ** 0.5)
                    image_embeds = image_embeds[:, 1:].view(-1, G, G, image_embeds.size(-1)) # remove [CLS], [B, G, G, C]
                    # record cumsum only, for fast mean pooling
                    image_embeds = image_embeds.cumsum(dim=1)
                    image_embeds = image_embeds.cumsum(dim=2)
                    image_embeds = F.pad(image_embeds, (0, 0, 1, 0, 1, 0), "constant", 0) # [B, 1+G, 1+G, C]
                    feature_shape = image_embeds.shape[1:]
                    # feature_shape = image_embeds.shape
                    for i, img_name in enumerate(img_names):
                        image_feat_dict[scan_name][img_name] = image_embeds[i].cpu()
        return image_feat_dict, feature_shape
   
    image_feat_dict, feature_shape = encode_feature(model, images)
    logging.info("Finished Pre-Computation")
    return image_feat_dict, grid_size

def reverse_augment(points, flip_x, flip_y, rot_mat, translation, no_rot=False):
    # points ~ [B, N_proposal, 8, 3]
    B, N_proposal, _, _ = points.shape
    # print(flip_x.shape, flip_y.shape, rot_mat.shape, translation.shape)
    # inverse translation
    # translation ~ [B, 3]
    translation = translation.view(-1, 1, 1, 3)
    points = points - translation
    
    if not no_rot:
        # inverse roation
        # rot_mat ~ [B, 3, 3]
        rot_mat = rot_mat.unsqueeze(1).expand(-1, N_proposal, -1, -1) # [B, N_propsal, 3, 3]
        # print(rot_mat.dtype, points.dtype)
        # NOTE: NEED CONFIRM!
        # (R^-1 P^T)^T
        points = torch.linalg.solve(rot_mat, points.transpose(-1, -2)).transpose(-1, -2) # => [B, N_propsal, 8, 3]
        # print(points.shape)

    # inverse flip
    # flip_x ~ [B]
    flip_x = flip_x.view(-1, 1, 1).expand(*points.shape[:-1]) # [B] => [B, N_propsal, 8]
    flip_y = flip_y.view(-1, 1, 1).expand(*points.shape[:-1]) # [B] => [B, N_propsal, 8]
    points[..., 0] = points[..., 0] * flip_x
    points[..., 1] = points[..., 1] * flip_y

    return points

def reverse_align(points, align_mat):
    B, N_proposal, _, _ = points.shape
    
    # inverse roation
    # align_mat ~ [B, 4, 4]
    align_mat = align_mat.unsqueeze(1).expand(-1, N_proposal, -1, -1) # [B, N_propsal, 4, 4]
    # print(rot_mat.dtype, points.dtype)
    # NOTE: NEED CONFIRM!
    # (R^-1 P^T)^T
    points = F.pad(points, (0, 1), mode="constant", value=1) # [B, N_propsal, 8, 4]
    points = torch.linalg.solve(align_mat, points.transpose(-1, -2)).transpose(-1, -2) # => [B, N_propsal, 8, 4]

    return points[...,0:3]

def reverse_align_simple(points, align_mat):
    B, P, _ = points.shape
    
    # inverse roation
    # align_mat ~ [B, 4, 4]
    # NOTE: NEED CONFIRM!
    # (R^-1 P^T)^T
    points = F.pad(points, (0, 1), mode="constant", value=1) # [B, P, 4]
    points = torch.linalg.solve(align_mat, points.transpose(-1, -2)).transpose(-1, -2) # => [B, P, 4]

    return points[...,0:3]



def rectangle_mean(prefix_sum, x1, y1, x2, y2):
    # Calculate the sum of the rectangle using the prefix sum array
    # if x2 > 1 and y2 > 1:
    #     sum_ = prefix_sum[x2-1][y2-1]
    # else:
    #     return zero

    # if x1 > 0:
    #     sum_ -= prefix_sum[x1-1][y2]
    # if y1 > 0:
    #     sum_ -= prefix_sum[x2][y1-1]
    # if x1 > 0 and y1 > 0:
    #     sum_ += prefix_sum[x1-1][y1-1]
    sum_ = prefix_sum[x2][y2] - prefix_sum[x1][y2] - prefix_sum[x2][y1] + prefix_sum[x1][y1]

    area = (y2 - y1) * (x2 - x1)

    return sum_ / area

def rectangle_mean_batch(prefix_sum, regions, eps=1e-8):
    r"""
    regions = [[x1, x2, y1, y2] ... [] [] ... ] , [N, 4]
    """
    # x1, x2, y1, y2 = regions.t() # [N]
    x1, x2, y1, y2 = regions
    N = x1.size(0)
    indices_x2y2 = torch.stack([torch.arange(N).to(x1), x2, y2])
    # print(indices_x2y2.shape)
    sum_ = prefix_sum[indices_x2y2.chunk(chunks=3,dim=0)]
    indices_x1y2 = torch.stack([torch.arange(N).to(x1), x1, y2])
    sum_ -= prefix_sum[indices_x1y2.chunk(chunks=3,dim=0)]
    indices_x2y1 = torch.stack([torch.arange(N).to(x1), x2, y1])
    sum_ -= prefix_sum[indices_x2y1.chunk(chunks=3,dim=0)]
    indices_x1y1 = torch.stack([torch.arange(N).to(x1), x1, y1])
    sum_ += prefix_sum[indices_x1y1.chunk(chunks=3,dim=0)]

    area = (y2 - y1) * (x2 - x1)

    # print(sum_.shape, area.shape)

    return (sum_.squeeze(0) + eps) / (area.unsqueeze(-1) + eps)

def get_region_feature_by_bbox(
    image_encoder_feat: torch.Tensor,
    grid_size: tuple,
    u: float,
    b: float,
    l: float,
    r: float,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""
    `image_encoder_feat`: (single) image feature output from ViT, [NC]
    `grid_size`: ViT encoder module's grid size, can be accessed by encoder.patch_embed.grid_size
    `u, b, l, r`: normalized x-y pos and h-w size (on screen-space)
    """
    image_encoder_feat = image_encoder_feat[1:]  # remove [CLS] # NOTE: DONE IN ENCODING
    H, W = grid_size
    C = image_encoder_feat.size(-1)
    image_encoder_feat = image_encoder_feat.view(-1, H, W, C) # NOTE: DONE IN ENCODING
    u = math.floor(u*(H-1)) # => [0...H-1]
    b = math.ceil(b*(H-1))
    l = math.floor(l*(W-1))
    r = math.ceil(r*(W-1))
    # u = math.ceil((y + h / 2) * H)
    # b = math.floor((y - h / 2) * H)
    # l = math.floor((x - w / 2) * W)
    # r = math.ceil((x + w / 2) * W)
    region_feature = image_encoder_feat[:, u : b + 1, l : r + 1]  # => [B, h, w, C]
    if reduction == "mean":
        region_feature = region_feature.mean(dim=1).mean(dim=1)  # => [B, C]
    return region_feature

def get_region_feature_by_bbox_batch(
    image_encoder_feat: torch.Tensor,
    grid_size: tuple,
    regions: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""
    `image_encoder_feat`: (batched) image feature output from ViT, [BNC]
    `grid_size`: ViT encoder module's grid size, can be accessed by encoder.patch_embed.grid_size
    `u, b, l, r`: normalized x-y pos and h-w size (on screen-space)
    """
    image_encoder_feat = image_encoder_feat[:, 1:]  # remove [CLS]
    B = image_encoder_feat.size(0)
    H, W = grid_size
    C = image_encoder_feat.size(-1)
    image_encoder_feat = image_encoder_feat.view(-1, H, W, C)
    regions[:, :2] *= (H-1)
    regions[:, 2:] *= (W-1)
    regions[:, 0].floor_()
    regions[:, 1].ceil_()
    regions[:, 2].floor_()
    regions[:, 3].ceil_()
    regions = regions.long().to(image_encoder_feat.device)

    grid_mask = torch.zeros([B, H, W, 1], dtype=torch.bool, device=image_encoder_feat.device)
    for i in range(B):
        grid_mask[i, regions[i,0]:regions[i,1]+1, regions[i,2]:regions[i,3]+1] = True

    region_feature = (region_feature * grid_mask).sum(dim=(1,2)) / grid_mask.sum(dim=(1,2,3)) # mean pooling
    return region_feature

def get_region_feature_by_bbox_batch_mpfast(
    image_encoder_feat: torch.Tensor,
    grid_size: tuple,
    regions: list[torch.Tensor],
) -> torch.Tensor:
    r"""
    `image_encoder_feat`: (batched) image feature output from ViT, [BNC]
    `grid_size`: ViT encoder module's grid size, can be accessed by encoder.patch_embed.grid_size
    `u, b, l, r`: normalized x-y pos and h-w size (on screen-space)
    Predefined mean pooling, use fast sum with pre-computed cumulative sum (instead of embedding)
    """
    # print(image_encoder_feat.shape)
    B = image_encoder_feat.size(0)
    H, W = grid_size
    assert H == W, "Only support square grid size"
    C = image_encoder_feat.size(-1)
    regions = [regions[i] * (H-1) for i in range(4)]
    ## regions[:, :2] *= (H-1)
    ## regions[:, 2:] *= (W-1)
    regions[0] = regions[0].floor().long().clamp(0, H-1)
    regions[1] = regions[1].ceil().long().clamp(0, H-1)
    regions[2] = regions[2].floor().long().clamp(0, W-1) # nan => -INT64_MIN => 0
    regions[3] = regions[3].ceil().long().clamp(0, W-1)

    empty_regions = ((regions[0] == regions[1]) | (regions[2] == regions[3]))
    full_regions = ((regions[0] == 0) & (regions[1] == H-1) & (regions[2] == 0) & (regions[3] == W-1))
    empty_regions = empty_regions | full_regions

    print((~empty_regions).sum().item())
    
    print([regions[i][:20] for i in range(4)])

    result = torch.zeros(B, C).to(image_encoder_feat)

    # for idx in range(B):
    #     # O(1)
    #     result[idx] = rectangle_mean(image_encoder_feat[idx], regions[idx][0], regions[idx][2], regions[idx][1], regions[idx][3])
    result = rectangle_mean_batch(image_encoder_feat, regions)
    # result = rectangle_mean_batch(image_encoder_feat, regions)


    return result, empty_regions

def parse_string(string):
    """
    Extracts 6 values from a string in the format "-?\d+\.\d+_(-?\d+\.\d+)_(-?\d+\.\d+)_(-?\d+)_(-?\d+)_(-?\d+)".
    The first 3 values are floats, and the last 3 are integers.

    Parameters:
        string (str): The input string to be parsed, e.g. "-1.87_1.74_1.82_1_1_0.jpg"


    Returns:
        list: A list of the extracted values, in the order they appear in the string. The values will be of the
            correct data types (float or int). If the string does not match the expected format, returns None.
    """

    # Use a regex pattern to extract the values from the input string
    pattern = r"(-?\d+\.\d+)_(-?\d+\.\d+)_(-?\d+\.\d+)_(-?\d+)_(-?\d+)_(-?\d+)"
    match = re.match(pattern, string)
    if match:
        # Convert the extracted values to the correct data types
        values = [float(val) for val in match.groups()[:3]] + [
            int(val) for val in match.groups()[3:]
        ]
        return values
    else:
        return None

def repeat_elements(lst, n):
    # Create a new list to store the repeated elements
    new_lst = []
    
    # Iterate over each element in the input list
    for element in lst:
        # Append the element to the new list n times
        new_lst.extend([element] * n)
    
    return new_lst

from pytorch3d.io import load_objs_as_meshes, load_obj, load_ply

from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    AmbientLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
)

def calculate_cube_corners(bbox):
    # Calculate the coordinates of the 8 corners of the cube
    # bbox: [..., 6], last dim = xyzhwl
    x = bbox[...,0]
    y = bbox[...,1]
    z = bbox[...,2]
    h = bbox[...,3]
    w = bbox[...,4]
    l = bbox[...,5]
    x_min = x - h / 2
    x_max = x + h / 2
    y_min = y - w / 2
    y_max = y + w / 2
    z_min = z - l / 2
    z_max = z + l / 2
    corners = [
        [x_min, y_min, z_min],
        [x_min, y_min, z_max],
        [x_min, y_max, z_min],
        [x_min, y_max, z_max],
        [x_max, y_min, z_min],
        [x_max, y_min, z_max],
        [x_max, y_max, z_min],
        [x_max, y_max, z_max],
    ]
    corners = [torch.stack(tensor, dim=-1) for tensor in corners] # => 8 [...,3]
    corners = torch.stack(corners, dim=-2) # => [...,8,3]
    return corners


def corners_to_edges(corners):
    # Calculate the endpoints of the 12 edges of the cube
    edges = [
        (corners[0], corners[1]),
        (corners[0], corners[2]),
        (corners[0], corners[4]),
        (corners[1], corners[3]),
        (corners[1], corners[5]),
        (corners[2], corners[3]),
        (corners[2], corners[6]),
        (corners[3], corners[7]),
        (corners[4], corners[5]),
        (corners[4], corners[6]),
        (corners[5], corners[7]),
        (corners[6], corners[7]),
    ]

    return edges


def contains(a, b):
    # Extract the coordinates of area A
    ax1, ax2, ay1, ay2 = a
    
    # Extract the coordinates of area B
    bx1, bx2, by1, by2 = b
    
    # Check if area B is completely inside area A
    return ax1 <= bx1 and ax2 >= bx2 and ay1 <= by1 and ay2 >= by2

def calculate_overlap(area_a, area_b):
    # Calculate the overlap in the x and y dimensions
    overlap_x = max(0, min(area_a[1], area_b[1]) - max(area_a[0], area_b[0]))
    overlap_y = max(0, min(area_a[3], area_b[3]) - max(area_a[2], area_b[2]))
    
    # Calculate the overlap area
    overlap_area = overlap_x * overlap_y

    if overlap_area == 0:
        return 0.
    
    # Calculate the area of area A
    area_a_area = (area_a[1] - area_a[0]) * (area_a[3] - area_a[2])

    if area_a_area == 0:
        return 0.
    
    # Calculate the relative overlap as the overlap area divided by the area of A
    relative_overlap = overlap_area / area_a_area
    
    return relative_overlap

def batch_iou(m, n):
    """
    Computes the Intersection over Union (IoU) of two series of 2D regions m and n, both in shape of [B, 4] 
    containing a batch of 4 corners.
    
    Args:
    - m: A PyTorch tensor of shape [B, 4] containing the coordinates of the corners of the first region.
    - n: A PyTorch tensor of shape [B, 4] containing the coordinates of the corners of the second region.
    
    Returns:
    - A PyTorch tensor of shape [B] containing the IoU of the two series of regions.
    """
    # Determine the coordinates of the intersection rectangle.
    x1 = torch.max(m[:, 0], n[:, 0])
    y1 = torch.max(m[:, 2], n[:, 2])
    x2 = torch.min(m[:, 1], n[:, 1])
    y2 = torch.min(m[:, 3], n[:, 3])

    # Compute the area of intersection rectangle.
    intersection_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # Compute the area of the two regions.
    m_area = (m[:, 1] - m[:, 0]) * (m[:, 3] - m[:, 2])
    n_area = (n[:, 1] - n[:, 0]) * (n[:, 3] - n[:, 2])

    # Compute the IoU.
    union_area = m_area + n_area - intersection_area
    iou = intersection_area / union_area

    return iou

def scene_bbox_to_2d_feat(
    bbox_3d=None, bbox_corners_3d=None, image_name=None, image_feat=None, device=None, grid_size=None, down=True, image_size=512
):
    r"""
    `bbox_3d`: [N, 6], first 3 dims are xyz, next 3 dims are hwl.
    `bbox_corners_3d`: [N, 8, 3], 8 corner's xyz.
    `image_name`: list of image_name, like "-1.87_1.74_1.82_1_1_0.jpg"
    `image_feat`: list of ViT image feature
    `grid_size`: ViT grid size
    `image_size`: scene image resolution
    `device`: renderer device
    
    Returns: region_feats, overlaps. overlaps can be determine whether to remain the object
    """
    assert bbox_3d is not None or bbox_corners_3d is not None, "Must provide bbox in xyzhwl or in 8 corners format"
    N = bbox_3d.size(0) if bbox_3d is not None else bbox_corners_3d.size(0)

    parsed = [parse_string(iname) for iname in image_name] # [B, 6]
    parsed = torch.tensor(parsed)
    print(parsed[0])
    eye = parsed[..., :3]
    dirs = parsed[..., 3:]
    if down:
        dirs[...,-1] += -0.5 # look downwards

    ## region_feats = []
    ## regions = torch.zeros((N, 4))
    # t0 = time()
    R, T = look_at_view_transform(at=eye + dirs, eye=eye, up=[[0, 0, 1] for _ in range(eye.shape[0])])
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    # print(f"get R,T,cameras: {time() - t0}") 

    
    if bbox_corners_3d is None:
        # corner_xyz = torch.tensor([calculate_cube_corners(*(bbox_3d[idx, :6].tolist())) for idx in range(N)]).to(device)
        corner_xyz = calculate_cube_corners(bbox_3d[:, :6])
    else:
        corner_xyz = bbox_corners_3d.to(device).float() # [B, 8, 3]
    
    # for idx in range(N):
        # R, T = look_at_view_transform(at=eye[None, idx] + dirs[None, idx], eye=eye[None, idx], up=[[0, 0, 1]])
        # cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    
    # t0 = time()
    corner_xy = cameras.transform_points_screen(
        corner_xyz,
        image_size=[[image_size, image_size]],
        with_xyflip=True,
    )[..., :2] # [B, 8, 2]
    # print(f"calculate corner transform: {time() - t0}") 
    # [B]
    xmax = corner_xy[..., 0].max(dim=-1).values# .clamp(0, image_size - 1) / (image_size - 1)  # [0...image_size-1] => [0...1]
    xmin = corner_xy[..., 0].min(dim=-1).values# .clamp(0, image_size - 1) / (image_size - 1)
    ymax = corner_xy[..., 1].max(dim=-1).values# .clamp(0, image_size - 1) / (image_size - 1)
    ymin = corner_xy[..., 1].min(dim=-1).values# .clamp(0, image_size - 1) / (image_size - 1)
    regions = torch.stack([xmin, xmax, ymin, ymax], dim=1)
    # nan proposal => set overlap to 0
    # proposal contains the whole image => set overlap to 0
    has_nan_bound = regions.isnan().any(dim=-1) # [B]
    overlaps = [
        calculate_overlap(regions[i].cpu().tolist(), (0, image_size, 0, image_size)) 
        # if not regions[i].isnan().any() and not contains(regions[i].cpu().tolist(), (0, image_size, 0, image_size)) 
        # else 0. 
        for i in range(N) 
    ]
    overlaps = torch.tensor(overlaps)
    overlaps[has_nan_bound] = 0.

    xmax = xmax.clamp(0, image_size - 1) / (image_size - 1)  # [0...image_size-1] => [0...1]
    xmin = xmin.clamp(0, image_size - 1) / (image_size - 1)
    ymax = ymax.clamp(0, image_size - 1) / (image_size - 1)
    ymin = ymin.clamp(0, image_size - 1) / (image_size - 1)
    
    region_feats, empty_mask = get_region_feature_by_bbox_batch_mpfast(
        image_feat, grid_size, [xmin, xmax, ymin, ymax]
    )
    overlaps[empty_mask] = 0.
    # print(f"inner get region_feats mpfast: {time() - t0}") 

    regions = torch.stack([xmin, xmax, ymin, ymax], dim=1) # [B, 4]

    return region_feats, overlaps, regions, corner_xy

# class VLMTarget:
#     def __init__(self, device, model_type: str='blip') -> None:
#         self.device=device
#         if model_type == 'blip':
#             self.image_feat_dict, self.grid_size = compute_blip_view_features()
#             self.image_size = 512
#             self.down = True
#         else:
#             raise NotImplementedError()

def jitter_bbox(corners: torch.Tensor, jitter_size: float=0.3) -> "torch.Tensor":
    # corners ~ [B, N_proposal, 8, 3] or [..., 8, 3]
    corner_size = corners.max(dim=-2) - corners.min(dim=-2) # [..., 3]
    corner_jitter = corner_size * torch.randn().clamp(-1, 1) * jitter_size
    corner_jitter = corner_jitter
    ...

def recover_from_cumsum(feat):
    # feat: [..., 1+G,1+G,C] cumsum
    # returns [..., G, G, C] rolled back features.
    ...

# @torch.no_grad()
# def weight_reset(m: torch.nn.Module):
#     # - check if the current module has reset_parameters & if it's callabed called it on m
#     reset_parameters = getattr(m, "reset_parameters", None)
#     if callable(reset_parameters):
#         m.reset_parameters()

    