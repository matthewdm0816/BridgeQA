""" 
Modified from: https://github.com/daveredrum/ScanRefer/blob/master/lib/loss_helper.py
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import torch.nn.functional as F

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from utils.nn_distance import nn_distance, huber_loss
from lib.ap_helper import parse_predictions
from lib.loss import SoftmaxRankingLoss
from utils.box_util import get_3d_box, get_3d_box_batch, box3d_iou, box3d_iou_batch
from icecream import ic

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3 # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8] # put larger weights on positive objectness

# class ChamferDistanceL2(torch.nn.Module):
#     f''' Chamder Distance L2
#     '''
#     def __init__(self, ignore_zeros=False):
#         super().__init__()
#         self.ignore_zeros = ignore_zeros

#     def forward(self, xyz1, xyz2):
#         batch_size = xyz1.size(0)
#         if batch_size == 1 and self.ignore_zeros:
#             non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
#             non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
#             xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
#             xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

#         dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
#         return torch.mean(dist1) + torch.mean(dist2)

def compute_vote_loss(data_dict):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        data_dict: dict (read-only)
    
    Returns:
        vote_loss: scalar Tensor
            
    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.

        Each seed point may vote for multiple translations v1,v2,v3
        A seed point may also be in the boxes of multiple objects:
        o1,o2,o3 with corresponding GT votes c1,c2,c3

        Then the loss for this seed point is:
            min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
    """

    # Load ground truth votes and assign them to seed points
    batch_size = data_dict['seed_xyz'].shape[0]
    num_seed = data_dict['seed_xyz'].shape[1] # B,num_seed,3
    vote_xyz = data_dict['vote_xyz'] # B,num_seed*vote_factor,3
    seed_inds = data_dict['seed_inds'].long() # B,num_seed in [0,num_points-1]

    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    seed_gt_votes_mask = torch.gather(data_dict['vote_label_mask'], 1, seed_inds)
    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3*GT_VOTE_FACTOR)
    seed_gt_votes = torch.gather(data_dict['vote_label'], 1, seed_inds_expand)
    seed_gt_votes += data_dict['seed_xyz'].repeat(1,1,3)

    # Compute the min of min of distance
    vote_xyz_reshape = vote_xyz.view(batch_size*num_seed, -1, 3) # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size*num_seed, GT_VOTE_FACTOR, 3) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1) # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = votes_dist.view(batch_size, num_seed)
    vote_loss = torch.sum(votes_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    return vote_loss

def compute_objectness_loss(data_dict):
    """ Compute objectness loss for the proposals.

    Args:
        data_dict: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        object_assignment: (batch_size, num_seed) Tensor with long int
            within [0,num_gt_object-1]
    """ 
    # Associate proposal and GT objects by point-to-point distances
    aggregated_vote_xyz = data_dict['aggregated_vote_xyz']
    gt_center = data_dict['center_label'][:,:,0:3]
    B = gt_center.shape[0]
    K = aggregated_vote_xyz.shape[1]
    K2 = gt_center.shape[1]
    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center) # dist1: BxK, dist2: BxK2

    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1 = torch.sqrt(dist1+1e-6)
    objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((B,K)).cuda()
    objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1>FAR_THRESHOLD] = 1

    # Compute objectness loss
    objectness_scores = data_dict['objectness_scores']
    #ic('os', objectness_scores.shape, objectness_label.shape)
    criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
    objectness_loss = criterion(objectness_scores.transpose(2,1), objectness_label)
    #ic('ol', objectness_loss.shape)
    #exit()
    objectness_loss = torch.sum(objectness_loss * objectness_mask)/(torch.sum(objectness_mask)+1e-6)

    # Set assignment
    object_assignment = ind1 # (B,K) with values in 0,1,...,K2-1

    return objectness_loss, objectness_label, objectness_mask, object_assignment


def compute_box_and_sem_cls_loss(data_dict, config):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        data_dict: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    object_assignment = data_dict['object_assignment']
    batch_size = object_assignment.shape[0]

    # Compute center loss
    pred_center = data_dict['center']
    gt_center = data_dict['center_label'][:,:,0:3]
    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
    box_label_mask = data_dict['box_label_mask']
    objectness_label = data_dict['objectness_label'].float()

    centroid_reg_loss1 = \
        torch.sum(dist1*objectness_label)/(torch.sum(objectness_label)+1e-6)
    centroid_reg_loss2 = \
        torch.sum(dist2*box_label_mask)/(torch.sum(box_label_mask)+1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2

    # Compute heading loss
    heading_class_label = torch.gather(data_dict['heading_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
    heading_class_loss = criterion_heading_class(data_dict['heading_scores'].transpose(2,1), heading_class_label) # (B,K)
    heading_class_loss = torch.sum(heading_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    heading_residual_label = torch.gather(data_dict['heading_residual_label'], 1, object_assignment) # select (B,K) from (B,K2)
    heading_residual_normalized_label = heading_residual_label / (np.pi/num_heading_bin)

    # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
    heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1], num_heading_bin).zero_()
    heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_heading_bin)
    heading_residual_normalized_loss = huber_loss(torch.sum(data_dict['heading_residuals_normalized']*heading_label_one_hot, -1) - heading_residual_normalized_label, delta=1.0) # (B,K)
    heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # Compute size loss
    size_class_label = torch.gather(data_dict['size_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_size_class = nn.CrossEntropyLoss(reduction='none')
    size_class_loss = criterion_size_class(data_dict['size_scores'].transpose(2,1), size_class_label) # (B,K)
    size_class_loss = torch.sum(size_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    size_residual_label = torch.gather(data_dict['size_residual_label'], 1, object_assignment.unsqueeze(-1).repeat(1,1,3)) # select (B,K,3) from (B,K2,3)
    size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
    size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_size_cluster)
    size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1,1,1,3) # (B,K,num_size_cluster,3)
    predicted_size_residual_normalized = torch.sum(data_dict['size_residuals_normalized']*size_label_one_hot_tiled, 2) # (B,K,3)

    mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0) # (1,1,num_size_cluster,3) 
    mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2) # (B,K,3)
    size_residual_label_normalized = size_residual_label / mean_size_label # (B,K,3)
    size_residual_normalized_loss = torch.mean(huber_loss(predicted_size_residual_normalized - size_residual_label_normalized, delta=1.0), -1) # (B,K,3) -> (B,K)
    size_residual_normalized_loss = torch.sum(size_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # 3.4 Semantic cls loss
    sem_cls_label = torch.gather(data_dict['sem_cls_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
    sem_cls_loss = criterion_sem_cls(data_dict['sem_cls_scores'].transpose(2,1), sem_cls_label) # (B,K)
    sem_cls_loss = torch.sum(sem_cls_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    return center_loss, heading_class_loss, heading_residual_normalized_loss, size_class_loss, size_residual_normalized_loss, sem_cls_loss


def compute_reference_loss(data_dict, config):
    """ Compute cluster reference loss
    Args:
        data_dict: dict (read-only)
    Returns:
        ref_loss, lang_loss, cluster_preds, cluster_labels
    """

    # unpack
    cluster_preds = data_dict["cluster_ref"] # B, num_proposal

    # predicted bbox
    pred_ref = data_dict['cluster_ref'].detach().cpu().numpy() # B, num_proposal
    pred_center = data_dict['center'].detach().cpu().numpy() # (B,K,3)
    pred_heading_class = torch.argmax(data_dict['heading_scores'], -1) # B,num_proposal
    pred_heading_residual = torch.gather(data_dict['heading_residuals'], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
    pred_heading_class = pred_heading_class.detach().cpu().numpy() # B,num_proposal
    pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal
    pred_size_class = torch.argmax(data_dict['size_scores'], -1) # B,num_proposal
    pred_size_residual = torch.gather(data_dict['size_residuals'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
    pred_size_class = pred_size_class.detach().cpu().numpy()
    pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal,3

    # ground truth bbox
    gt_center = data_dict['ref_center_label'].cpu().numpy() # (B,3)
    gt_heading_class = data_dict['ref_heading_class_label'].cpu().numpy() # B
    gt_heading_residual = data_dict['ref_heading_residual_label'].cpu().numpy() # B
    gt_size_class = data_dict['ref_size_class_label'].cpu().numpy() # B
    gt_size_residual = data_dict['ref_size_residual_label'].cpu().numpy() # B,3
    # convert gt bbox parameters to bbox corners
    gt_obb_batch = config.param2obb_batch(gt_center[:, 0:3], gt_heading_class, gt_heading_residual,
                    gt_size_class, gt_size_residual)
    gt_bbox_batch = get_3d_box_batch(gt_obb_batch[:, 3:6], gt_obb_batch[:, 6], gt_obb_batch[:, 0:3])

    # compute the iou score for all predictd positive ref
    batch_size, num_proposals = cluster_preds.shape
    labels = np.zeros((batch_size, num_proposals))

    for i in range(pred_ref.shape[0]):
        # convert the bbox parameters to bbox corners
        pred_obb_batch = config.param2obb_batch(pred_center[i, :, 0:3], pred_heading_class[i], pred_heading_residual[i],
                    pred_size_class[i], pred_size_residual[i])
        pred_bbox_batch = get_3d_box_batch(pred_obb_batch[:, 3:6], pred_obb_batch[:, 6], pred_obb_batch[:, 0:3])
        ious = box3d_iou_batch(pred_bbox_batch, np.tile(gt_bbox_batch[i], (num_proposals, 1, 1)))
        labels[i, ious.argmax()] = 1 # treat the bbox with highest iou score as the gt

    cluster_labels = torch.FloatTensor(labels).cuda() # batch_size, num_proposal
    # reference loss
    criterion_ref = SoftmaxRankingLoss()
    loss_ref = criterion_ref(cluster_preds, cluster_labels.float().clone(), mask=data_dict["ref_obj_mask"])
    return loss_ref, cluster_preds, cluster_labels


def compute_lang_classification_loss(data_dict):
    loss_lang = F.cross_entropy(data_dict["lang_scores"], data_dict["object_cat"], reduction='none')
    mask = data_dict["ref_obj_mask"] + 1e-8
    loss_lang = torch.sum(loss_lang * mask) / torch.sum(mask)
    return loss_lang


def compute_answer_classification_loss(data_dict):
    """ Compute cluster reference loss

    Args:
        data_dict: dict (read-only)

    Returns:
        ref_loss, lang_loss, cluster_preds, cluster_labels
    """
    loss_answer = None
    # decoder/main branch losses
    if "decoder_loss" in data_dict:
        loss_answer = data_dict["decoder_loss"] # already computed from text decoder
    elif False and "answer_cat_scores" in data_dict:
        #  data_dict["answer_cat_scores"]: batch_size, num_answers
        loss_answer = F.binary_cross_entropy_with_logits(data_dict["answer_scores"], data_dict["answer_cat_scores"], reduction='sum') / data_dict["answer_scores"].shape[0]
    else:
        loss_answer = F.cross_entropy(data_dict["answer_scores"], data_dict["answer_cat"])
    
    # 3D scene loss
    if "answer_scores_scene" in data_dict:
        if False and "answer_cat_scores" in data_dict:
            #  data_dict["answer_cat_scores"]: batch_size, num_answers
            loss_answer += F.binary_cross_entropy_with_logits(data_dict["answer_scores_scene"], data_dict["answer_cat_scores"], reduction='sum') / data_dict["answer_scores_scene"].shape[0]
        else:
            loss_answer += F.cross_entropy(data_dict["answer_scores_scene"], data_dict["answer_cat"])

    # 2D3D loss
    if "answer_scores_2d3d" in data_dict:
        if False and "answer_cat_scores" in data_dict:
            #  data_dict["answer_cat_scores"]: batch_size, num_answers
            loss_answer += F.binary_cross_entropy_with_logits(data_dict["answer_scores_scene"], data_dict["answer_cat_scores"], reduction='sum') / data_dict["answer_scores_scene"].shape[0]
        else:
            loss_answer += F.cross_entropy(data_dict["answer_scores_2d3d"], data_dict["answer_cat"])

    return loss_answer

def kl_divergence(mu, logvar):
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kld.sum(dim=-1).mean()

def kl_divergence2(mu1, mu2, logvar1, logvar2):
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    kld = 0.5 * (logvar2 - logvar1 + (var1 + (mu1 - mu2).pow(2)) / var2 - 1)
    return kld.sum(dim=-1).mean()

import torch.nn.functional as F

def kl_divergence_logits(p, q, softmaxed=False, temp=5):
    """
    Computes the KL divergence between two probability vectors p and q.
    
    Args:
    - p: A PyTorch tensor of shape [B, N] containing the probabilities of B events across N categories.
    - q: A PyTorch tensor of shape [B, N] containing the probabilities of B events across N categories.
    
    Returns:
    - A PyTorch tensor of shape [B] containing the KL divergence between p and q.
    """
    # Apply log-softmax to p and q along the second dimension
    if not softmaxed:
        p = F.softmax(p / temp, dim=1)
        q = F.softmax(q / temp, dim=1)

    # logp = p.log()
    # logq = q.log()
    # avoid 0/0 nan result
    p = p.clamp(min=1e-8)
    q = q.clamp(min=1e-8)
    
    # Compute the elementwise product of p and the difference between logp and logq
    diff = -p * q.log()
    
    # Sum over the second dimension and negate to get the KL divergence
    kl = torch.sum(diff, dim=1)
    
    return kl

import torch

def js_divergence_logits(p, q):
    """
    Computes the Jensen-Shannon divergence between two probability vectors p and q.
    
    Args:
    - p: A PyTorch tensor of shape [B, N] containing the probabilities of B events across N categories.
    - q: A PyTorch tensor of shape [B, N] containing the probabilities of B events across N categories.
    
    Returns:
    - A PyTorch tensor of shape [B] containing the JS divergence between p and q.
    """
    p = F.softmax(p, dim=1)
    q = F.softmax(q, dim=1)
    # Compute the midpoint between p and q
    m = 0.5 * (p + q)
    
    # Compute the KL divergence between p and m
    kl_pm = kl_divergence_logits(p, m, softmaxed=True)
    
    # Compute the KL divergence between q and m
    kl_qm = kl_divergence_logits(q, m, softmaxed=True)
    
    # Compute the JS divergence as the average of the two KL divergences
    js = 0.5 * (kl_pm + kl_qm)
    
    return js



def compute_align_loss(data_dict: dict):
    # ic(data_dict["object_feat_ts"].shape, data_dict["region_feats"].shape)
    object_feat_ts = data_dict["object_feat_ts"] # [B, N_proposal, F]
    region_feats= data_dict["region_feats"] # [B, N_proposal, F]
    align_mask = data_dict["align_mask"] # [B, N_proposal]
    if "region_feats_recon" not in data_dict: 
        # non-variational matching
        single_similarity = F.cosine_similarity(object_feat_ts[align_mask], region_feats[align_mask], dim=-1) # [N_aligned_p]
        loss_align = 1 - single_similarity.mean()
        ic(loss_align.item())
        # loss_align = F.mse_loss(data_dict["object_feat_ts"], data_dict["region_feats"], reduction='none').sum(dim=-1).mean()
        # loss_align *= data_dict["align_bs"]
        if torch.isnan(loss_align):
            loss_align = torch.tensor(0.0).to(loss_align) # no aligned regions (0/0 makes nan)
    else:
        # vae loss
        loss_recon = (
            F.mse_loss(object_feat_ts[align_mask], data_dict["object_feats_recon"]) +
            F.mse_loss(region_feats[align_mask], data_dict["region_feats_recon"])
        )
        # loss_prior = (
        #     torch.mean(-0.5 * torch.sum(1 + data_dict["object_logvar"] - data_dict["object_mu"].pow(2) - data_dict["object_logvar"].exp(), dim = -1)) + 
        #     torch.mean(-0.5 * torch.sum(1 + data_dict["region_logvar"] - data_dict["region_mu"].pow(2) - data_dict["region_logvar"].exp(), dim = -1))
        # )
        loss_prior = kl_divergence(data_dict["object_mu"], data_dict["object_logvar"]) + kl_divergence(data_dict["region_mu"], data_dict["region_logvar"])
        loss_vae = loss_recon + 0.1 * loss_prior
        # variational align loss
        # loss_align = -0.5 * (
        #     data_dict["region_logvar"] - data_dict["object_logvar"] + 
        #     (data_dict["object_logvar"].exp() + (data_dict["object_mu"] - data_dict["region_mu"]).pow(2)) / data_dict["region_logvar"].exp()
        #     -1
        # ).sum(dim=-1).mean()
        loss_align = kl_divergence2(data_dict["object_mu"], data_dict["region_mu"], data_dict["object_logvar"], data_dict["region_logvar"])
        ic(loss_align, loss_recon, loss_prior)
        if torch.isnan(loss_align):
            loss_align = torch.tensor(0.0).to(loss_align) # no aligned regions (0/0 makes nan)
        loss_align += loss_vae
        loss_align *= data_dict["align_bs"]
        # loss_align = loss_vae
        

    if data_dict.get("use_contrastive", False):
        # TODO: in GT-matched feature, contrastive within a category!
        # intra-image contrastive
        # import torch.linalg
        ic("Computing contrastive loss...")
        temp = data_dict.get("contrastive_temperature", 10)
        B, N_proposal, _ = object_feat_ts.shape
        region_feats[~align_mask] = 0. # avoid nan

        with torch.no_grad():
            sim_mask = torch.zeros(B, N_proposal, N_proposal).to(align_mask)
            sim_mask[~align_mask] = True
            sim_mask = sim_mask.transpose(-1, -2)
            sim_mask[~align_mask] = True
            sim_mask = sim_mask.transpose(-1, -2)
            targets = torch.arange(N_proposal, device=object_feat_ts.device).unsqueeze(0).expand(B, -1)

        similarity = F.cosine_similarity(object_feat_ts[..., None, :, :], region_feats[..., :, None, :], dim=-1)
        ic(similarity.shape, targets.shape)
        

        # object_feat_ts_norm = F.normalize(object_feat_ts, dim=-1) # [B, N_proposal, F]
        ## region_feats_norm = F.normalize(region_feats, dim=-1) # [B, N_proposal, F]
        # similarity = torch.bmm(object_feat_ts_norm, region_feats_norm.transpose(-1, -2)) # [B, N_proposal, N_proposal]
        # similarity[~align_mask] = -1e4
        # similarity = similarity.transpose(-1, -2)
        # similarity[~align_mask] = -1e4
        # similarity = similarity.transpose(-1, -2)
        similarity[sim_mask] = -1e6
        # similarity = (similarity * T)
        loss_intra = F.cross_entropy((similarity * temp).transpose(-1, -2), targets, reduction="none")
        loss_intra = loss_intra[align_mask].mean()
        ic(loss_intra.item())

        # inter-image contrastive
        # perm = torch.randperm(B, device=object_feat_ts.device)
        # perm_mask = (perm != torch.arange(B, device=perm.device)) # 忽略没动的sample
        ## region_feats_norm_perm = region_feats_norm[perm] # [B, N_proposal, F]
        ## region_feats_norm_perm = region_feats_norm[perm] # [B, N_proposal, F]
        with torch.no_grad():
            sim_mask = torch.zeros(N_proposal, B, B).to(align_mask)
            sim_mask[~align_mask.transpose(0, 1)] = True
            sim_mask = sim_mask.transpose(-1, -2)
            sim_mask[~align_mask.transpose(0, 1)] = True
            sim_mask = sim_mask.transpose(-1, -2)
            targets_batch = torch.arange(B, device=object_feat_ts.device).unsqueeze(0).expand(N_proposal, -1)

        similarity_batch = F.cosine_similarity(object_feat_ts.transpose(0,1)[..., None, :, :], region_feats.transpose(0,1)[..., :, None, :], dim=-1)
        # similarity_batch = torch.bmm(object_feat_ts_norm.transpose(0, 1), region_feats_norm.permute(1, 2, 0)) # [N_proposal, B, B]
        # similarity_batch[~align_mask.transpose(0, 1)] = -1e4
        # similarity_batch = similarity_batch.transpose(-1, -2)
        # similarity_batch[~align_mask.transpose(0, 1)] = -1e4
        # similarity_batch = similarity_batch.transpose(-1, -2)
        similarity_batch[sim_mask] = -1e6
        loss_inter = F.cross_entropy((similarity_batch * temp).transpose(-1, -2), targets_batch, reduction="none") # [N_proposal, B]
        loss_inter = loss_inter.transpose(0, 1)[align_mask].mean()
        ic(loss_inter.item())

        ic(similarity.isnan().any())
        ic(similarity_batch.isnan().any())
        ic(object_feat_ts.isnan().any())
        ic(region_feats.isnan().any())

        loss_align += loss_intra + loss_inter

    if "blip_fused_feat" in data_dict:
        loss_align_fused = 1 - F.cosine_similarity(data_dict["blip_fused_feat"].mean(dim=1), data_dict["fuse_feat"], dim=-1).sum()
        loss_align += loss_align_fused
    # ic(loss_align.detach().cpu().item())

    if "soft_label" in data_dict:
        # loss_align_label = js_divergence_logits(data_dict["answer_scores"], data_dict["soft_label"]).sum()
        loss_align_label = kl_divergence_logits(data_dict["answer_scores"], data_dict["soft_label"], temp=1, softmaxed=False).sum()
        ic(f"loss_align_label: {loss_align_label.item()}")
        loss_align += 0.01 * loss_align_label
    
    return loss_align

def compute_align_loss_gt_matched(data_dict):
    r"""
    "objectness_label" ~ pred obj is valid
    "box_label_mask" ~ gt bbox padding mask
    'center_label' ~ gt bbox center
    'object_assignment' ~ [B, K] assigns each proposal to its closest label
    """
    

    return ...

def compute_selector_loss(data_dict, loss):
    # TODO
    F.mse_loss(data_dict["selector_score"], loss)
    return data_dict["selector_score"]


def get_loss(data_dict, config, detection=True, use_reference=True, use_lang_classifier=False, 
        use_answer=True, loss_weights=None, use_vlm_align=False, use_selector=False, use_gt_obj_align=False,
    ):
    """ Loss functions

    Args:
        data_dict: dict
        config: dataset config instance
        reference: flag (False/True)
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """
    if loss_weights is None:
        loss_weights = {}

    # Vote loss
    vote_loss = compute_vote_loss(data_dict)

    # Obj loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = compute_objectness_loss(data_dict)
    num_proposal = objectness_label.shape[1]
    total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
    data_dict['objectness_label'] = objectness_label
    data_dict['objectness_mask'] = objectness_mask
    data_dict['object_assignment'] = object_assignment
    data_dict['pos_ratio'] = torch.sum(objectness_label.float().cuda())/float(total_num_proposal)
    data_dict['neg_ratio'] = torch.sum(objectness_mask.float())/float(total_num_proposal) - data_dict['pos_ratio']

    # Box loss and sem cls loss
    center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss = compute_box_and_sem_cls_loss(data_dict, config)
    box_loss = center_loss + 0.1 * heading_cls_loss + heading_reg_loss + 0.1 * size_cls_loss + size_reg_loss

    if detection:
        data_dict['vote_loss'] = vote_loss
        data_dict['objectness_loss'] = objectness_loss
        data_dict['center_loss'] = center_loss
        data_dict['heading_cls_loss'] = heading_cls_loss
        data_dict['heading_reg_loss'] = heading_reg_loss
        data_dict['size_cls_loss'] = size_cls_loss
        data_dict['size_reg_loss'] = size_reg_loss
        data_dict['sem_cls_loss'] = sem_cls_loss
        data_dict['box_loss'] = box_loss
    else:
        data_dict['vote_loss'] = torch.zeros(1)[0].cuda()
        data_dict['objectness_loss'] = torch.zeros(1)[0].cuda()
        data_dict['center_loss'] = torch.zeros(1)[0].cuda()
        data_dict['heading_cls_loss'] = torch.zeros(1)[0].cuda()
        data_dict['heading_reg_loss'] = torch.zeros(1)[0].cuda()
        data_dict['size_cls_loss'] = torch.zeros(1)[0].cuda()
        data_dict['size_reg_loss'] = torch.zeros(1)[0].cuda()
        data_dict['sem_cls_loss'] = torch.zeros(1)[0].cuda()
        data_dict['box_loss'] = torch.zeros(1)[0].cuda()

    if use_reference:
        # Reference loss
        ref_loss, _, cluster_labels = compute_reference_loss(data_dict, config)
        data_dict["cluster_labels"] = cluster_labels
        data_dict["ref_loss"] = ref_loss
    else:
        # Reference loss
        data_dict["cluster_labels"] = objectness_label.new_zeros(objectness_label.shape).cuda()
        data_dict["cluster_ref"] = objectness_label.new_zeros(objectness_label.shape).float().cuda()
        # store
        data_dict["ref_loss"] = torch.zeros(1)[0].cuda()

    if use_answer:
        data_dict["answer_loss"] = compute_answer_classification_loss(data_dict)
    else:
        data_dict["answer_loss"] = torch.zeros(1)[0].cuda()


    #if reference and use_lang_classifier:
    if use_lang_classifier:
        data_dict["lang_loss"] = compute_lang_classification_loss(data_dict)
    else:
        data_dict["lang_loss"] = torch.zeros(1)[0].cuda()

    if use_vlm_align:
        # if use_gt_obj_align:
        #     data_dict["align_loss"] = compute_align_loss_gt_matched(data_dict)
        # else:
        data_dict["align_loss"] = compute_align_loss(data_dict)
    else:
        data_dict["align_loss"] = torch.zeros(1)[0].cuda()


    loss = loss_weights.get('vote_loss', 1.) * data_dict['vote_loss'] \
            + loss_weights.get('objectness_loss', 1.) * data_dict['objectness_loss'] \
            + loss_weights.get('box_loss', 1.) * data_dict['box_loss'] \
            + loss_weights.get('sem_cls_loss', 1.) * data_dict['sem_cls_loss'] \
            + loss_weights.get('ref_loss', 1.) * data_dict["ref_loss"] \
            + loss_weights.get('lang_loss', 1.) * data_dict["lang_loss"] \
            + loss_weights.get('answer_loss', 1.) * data_dict['answer_loss'] \
            + loss_weights.get('align_loss', 1.) * data_dict['align_loss'] \
            + loss_weights.get('mae_loss', 1.) * data_dict['mae_loss'] \
    
    # if use_selector:
    #     data_dict["selector_loss"] = compute_selector_loss(data_dict, loss)
    # else:
    #     data_dict["selector_loss"] = torch.zeros(1)[0].cuda()

    # loss += loss_weights.get('align_loss', 1.) * data_dict['align_loss'] \
    #     + loss_weights.get('selector_loss', 1.) * data_dict['selector_loss']  

    loss *= 10 # amplify
    data_dict['loss'] = loss
    return loss, data_dict


