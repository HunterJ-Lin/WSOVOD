# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
from typing import List, Tuple, Union
import torch

from detectron2.layers import batched_nms, cat, move_device_like
from detectron2.modeling.poolers import convert_boxes_to_pooler_format
from detectron2.structures import Boxes, Instances
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.events import get_event_storage
from wsovod.layers import csc

logger = logging.getLogger(__name__)


def _is_tracing():
    # (fixed in TORCH_VERSION >= 1.9)
    if torch.jit.is_scripting():
        # https://github.com/pytorch/pytorch/issues/47379
        return False
    else:
        return torch.jit.is_tracing()


def find_top_rpn_proposals(
    proposals: List[torch.Tensor],
    pred_objectness_logits: List[torch.Tensor],
    image_sizes: List[Tuple[int, int]],
    nms_thresh: float,
    pre_nms_topk: int,
    post_nms_topk: int,
    min_box_size: float,
    training: bool,
):
    """
    For each feature map, select the `pre_nms_topk` highest scoring proposals,
    apply NMS, clip proposals, and remove small boxes. Return the `post_nms_topk`
    highest scoring proposals among all the feature maps for each image.

    Args:
        proposals (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A, 4).
            All proposal predictions on the feature maps.
        pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A).
        image_sizes (list[tuple]): sizes (h, w) for each image
        nms_thresh (float): IoU threshold to use for NMS
        pre_nms_topk (int): number of top k scoring proposals to keep before applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is per
            feature map.
        post_nms_topk (int): number of top k scoring proposals to keep after applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is total,
            over all feature maps.
        min_box_size (float): minimum proposal box side length in pixels (absolute units
            wrt input images).
        training (bool): True if proposals are to be used in training, otherwise False.
            This arg exists only to support a legacy bug; look for the "NB: Legacy bug ..."
            comment.

    Returns:
        list[Instances]: list of N Instances. The i-th Instances
            stores post_nms_topk object proposals for image i, sorted by their
            objectness score in descending order.
    """
    num_images = len(image_sizes)
    device = (
        proposals[0].device
        if torch.jit.is_scripting()
        else ("cpu" if torch.jit.is_tracing() else proposals[0].device)
    )

    # 1. Select top-k anchor for every level and every image
    topk_scores = []  # #lvl Tensor, each of shape N x topk
    topk_proposals = []
    level_ids = []  # #lvl Tensor, each of shape (topk,)
    batch_idx = move_device_like(torch.arange(num_images, device=device), proposals[0])
    for level_id, (proposals_i, logits_i) in enumerate(zip(proposals, pred_objectness_logits)):
        Hi_Wi_A = logits_i.shape[1]
        if isinstance(Hi_Wi_A, torch.Tensor):  # it's a tensor in tracing
            num_proposals_i = torch.clamp(Hi_Wi_A, max=pre_nms_topk)
        else:
            num_proposals_i = min(Hi_Wi_A, pre_nms_topk)

        # sort is faster than topk: https://github.com/pytorch/pytorch/issues/22812
        # topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)
        logits_i, idx = logits_i.sort(descending=True, dim=1)
        topk_scores_i = logits_i.narrow(1, 0, num_proposals_i)
        topk_idx = idx.narrow(1, 0, num_proposals_i)

        # each is N x topk
        topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # N x topk x 4

        topk_proposals.append(topk_proposals_i)
        topk_scores.append(topk_scores_i)
        level_ids.append(
            move_device_like(
                torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device),
                proposals[0],
            )
        )

    # 2. Concat all levels together
    topk_scores = cat(topk_scores, dim=1)
    topk_proposals = cat(topk_proposals, dim=1)
    level_ids = cat(level_ids, dim=0)

    # 3. For each image, run a per-level NMS, and choose topk results.
    results: List[Instances] = []
    for n, image_size in enumerate(image_sizes):
        boxes = Boxes(topk_proposals[n])
        scores_per_img = topk_scores[n]
        lvl = level_ids

        valid_mask = torch.isfinite(boxes.tensor).all(dim=1) & torch.isfinite(scores_per_img)
        if not valid_mask.all():
            if training:
                raise FloatingPointError(
                    "Predicted boxes or scores contain Inf/NaN. Training has diverged."
                )
            boxes = boxes[valid_mask]
            scores_per_img = scores_per_img[valid_mask]
            lvl = lvl[valid_mask]
        boxes.clip(image_size)

        # filter empty boxes
        keep = boxes.nonempty(threshold=min_box_size)
        if _is_tracing() or keep.sum().item() != len(boxes):
            boxes, scores_per_img, lvl = boxes[keep], scores_per_img[keep], lvl[keep]

        keep = batched_nms(boxes.tensor, scores_per_img, lvl, nms_thresh)
        # In Detectron1, there was different behavior during training vs. testing.
        # (https://github.com/facebookresearch/Detectron/issues/459)
        # During training, topk is over the proposals from *all* images in the training batch.
        # During testing, it is over the proposals for each image separately.
        # As a result, the training behavior becomes batch-dependent,
        # and the configuration "POST_NMS_TOPK_TRAIN" end up relying on the batch size.
        # This bug is addressed in Detectron2 to make the behavior independent of batch size.
        keep = keep[:post_nms_topk]  # keep is already sorted

        res = Instances(image_size)
        res.proposal_boxes = boxes[keep]
        res.objectness_logits = scores_per_img[keep]

        results.append(res)
    return results


def find_top_rpn_proposals_group(
    proposals: List[torch.Tensor],
    pred_objectness_logits: List[torch.Tensor],
    image_sizes: List[Tuple[int, int]],
    num_anchors: List[int],
    nms_thresh: float,
    pre_nms_topk: int,
    post_nms_topk: int,
    min_box_size: float,
    training: bool,
    cpgs: torch.Tensor = None,
    cpg_strides: Tuple[int] = None,
):
    """
    For each feature map, select the `pre_nms_topk` highest scoring proposals,
    apply NMS, clip proposals, and remove small boxes. Return the `post_nms_topk`
    highest scoring proposals among all the feature maps for each image.

    Args:
        proposals (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A, 4).
            All proposal predictions on the feature maps.
        pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A).
        image_sizes (list[tuple]): sizes (h, w) for each image
        nms_thresh (float): IoU threshold to use for NMS
        pre_nms_topk (int): number of top k scoring proposals to keep before applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is per
            feature map.
        post_nms_topk (int): number of top k scoring proposals to keep after applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is total,
            over all feature maps.
        min_box_size (float): minimum proposal box side length in pixels (absolute units
            wrt input images).
        training (bool): True if proposals are to be used in training, otherwise False.
            This arg exists only to support a legacy bug; look for the "NB: Legacy bug ..."
            comment.

    Returns:
        list[Instances]: list of N Instances. The i-th Instances
            stores post_nms_topk object proposals for image i, sorted by their
            objectness score in descending order.
    """
    num_images = len(image_sizes)
    device = proposals[0].device

    num_pre_nms = [0, 0, 0, 0]
    num_post_nms = 0

    # 1. Select top-k anchor for every level and every image
    topk_scores = []  # #lvl Tensor, each of shape N x topk
    topk_proposals = []
    level_ids = []  # #lvl Tensor, each of shape (topk,)
    batch_idx = torch.arange(num_images, device=device)
    for level_id, (proposals_i, logits_i) in enumerate(zip(proposals, pred_objectness_logits)):
        Hi_Wi_A = logits_i.shape[1]
        Hi_Wi = int(Hi_Wi_A / num_anchors[level_id])
        logits_i = logits_i.view(-1, Hi_Wi, num_anchors[level_id])
        proposals_i = proposals_i.view(-1, Hi_Wi, num_anchors[level_id], 4)
        num_pre_nms[0] += Hi_Wi_A
        for anchor_id in range(num_anchors[level_id]):
            if isinstance(Hi_Wi, torch.Tensor):  # it's a tensor in tracing
                num_proposals_i_a = torch.clamp(Hi_Wi, max=pre_nms_topk)
            else:
                num_proposals_i_a = min(Hi_Wi, pre_nms_topk)

            logits_i_a = logits_i[:, :, anchor_id]
            proposals_i_a = proposals_i[:, :, anchor_id, :]

            if True and False:
                width_i_a = proposals_i_a[..., 2] - proposals_i_a[..., 0]
                height_i_a = proposals_i_a[..., 3] - proposals_i_a[..., 1]
                size_i_a = width_i_a * height_i_a
                aspect_i_a = width_i_a / height_i_a
                print(device, size_i_a.sqrt().mean(), size_i_a.mean().sqrt(), aspect_i_a.mean())

            # sort is faster than topk (https://github.com/pytorch/pytorch/issues/22812)
            # topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)
            logits_i_a, idx = logits_i_a.sort(descending=True, dim=1)
            topk_scores_i_a = logits_i_a.narrow(1, 0, num_proposals_i_a)
            topk_idx = idx.narrow(1, 0, num_proposals_i_a)

            # each is N x topk
            topk_proposals_i_a = proposals_i_a[batch_idx[:, None], topk_idx]  # N x topk x 4

            topk_proposals.append(topk_proposals_i_a)
            topk_scores.append(topk_scores_i_a)
            level_ids.append(
                torch.full(
                    (num_proposals_i_a,),
                    level_id * 1000 + anchor_id,
                    dtype=torch.int64,
                    device=device,
                )
            )

    # 2. Concat all levels together
    topk_scores = cat(topk_scores, dim=1)
    topk_proposals = cat(topk_proposals, dim=1)
    level_ids = cat(level_ids, dim=0)

    # 3. For each image, run a per-level NMS, and choose topk results.
    results: List[Instances] = []
    for n, image_size in enumerate(image_sizes):
        boxes = Boxes(topk_proposals[n])
        scores_per_img = topk_scores[n]
        lvl = level_ids

        num_pre_nms[1] += len(boxes)

        valid_mask = torch.isfinite(boxes.tensor).all(dim=1) & torch.isfinite(scores_per_img)
        if not valid_mask.all():
            if training:
                raise FloatingPointError(
                    "Predicted boxes or scores contain Inf/NaN. Training has diverged."
                )
            boxes = boxes[valid_mask]
            scores_per_img = scores_per_img[valid_mask]
            lvl = lvl[valid_mask]
        boxes.clip(image_size)

        # filter empty boxes
        keep = boxes.nonempty(threshold=min_box_size)
        if _is_tracing() or keep.sum().item() != len(boxes):
            boxes, scores_per_img, lvl = boxes[keep], scores_per_img[keep], lvl[keep]

        num_pre_nms[2] += len(boxes)
        if isinstance(cpgs, List) and isinstance(cpgs[0], torch.Tensor):
            # print("doing lmap constraint")

            labels = torch.ones((1, 1), dtype=cpgs[n].dtype, device=cpgs[n].device)
            preds = torch.ones((1, 1), dtype=cpgs[n].dtype, device=cpgs[n].device)
            rois = convert_boxes_to_pooler_format([boxes])
            rois[:, 1:] = rois[:, 1:] / cpg_strides[n]
            W, PL, NL = csc(
                cpgs[n].unsqueeze(0).unsqueeze(0),
                labels,
                preds,
                rois,
                0.7,
                False,
                0.1,
                0.2,
                0.0,
                True,
                1.8,
            )

            # valid_mask = W.squeeze() >= 0
            # print(W.shape, rois.shape, valid_mask.sum())

            # boxes = boxes[valid_mask]
            # scores_per_img = scores_per_img[valid_mask]
            # lvl = lvl[valid_mask]

            # print(scores_per_img.shape, W.shape)
            # print(scores_per_img.shape, (W+1.0).shape)
            scores_per_img = scores_per_img * (W.squeeze() + 1.0)

            # cpg = cpgs[n] > 0.1
            # cpg = torch.nonzero(cpg, as_tuple=False)
            # # print(cpg)
            # valid_masks = None
            # for i in range(cpg.size(0)):
            # y = cpg[i, 0] * cpg_strides[n]
            # x = cpg[i, 1] * cpg_strides[n]
            # # y = cpg[i, 0]
            # # x = cpg[i, 1]

            # valid_mask = (
            # (boxes.tensor[:, 0] < x)
            # & (boxes.tensor[:, 1] < y)
            # & (boxes.tensor[:, 2] > x)
            # & (boxes.tensor[:, 3] > y)
            # )
            # if valid_masks is not None:
            # valid_masks = valid_masks | valid_mask
            # else:
            # valid_masks = valid_mask
            # # print(torch.sum(valid_masks))

            # if valid_masks is not None:
            # boxes = boxes[valid_masks]
            # scores_per_img = scores_per_img[valid_masks]
            # lvl = lvl[valid_masks]

        # print("find_top_rpn_proposals_group", len(boxes))

        num_pre_nms[3] += len(boxes)

        keep = batched_nms(boxes.tensor, scores_per_img, lvl, nms_thresh)
        num_post_nms += keep.numel()
        # In Detectron1, there was different behavior during training vs. testing.
        # (https://github.com/facebookresearch/Detectron/issues/459)
        # During training, topk is over the proposals from *all* images in the training batch.
        # During testing, it is over the proposals for each image separately.
        # As a result, the training behavior becomes batch-dependent,
        # and the configuration "POST_NMS_TOPK_TRAIN" end up relying on the batch size.
        # This bug is addressed in Detectron2 to make the behavior independent of batch size.
        keep = keep[:post_nms_topk]  # keep is already sorted
        # if keep.numel() > post_nms_topk:
        # keep = torch.multinomial(scores_per_img[keep] - scores_per_img[keep].min(), post_nms_topk)

        res = Instances(image_size)
        res.proposal_boxes = boxes[keep]
        res.objectness_logits = scores_per_img[keep]
        res.level_ids = lvl[keep]
        results.append(res)

    if training:
        storage = get_event_storage()
        storage.put_scalar("rpn/num_proposals_pre_nms_0", num_pre_nms[0] / num_images)
        storage.put_scalar("rpn/num_proposals_pre_nms_1", num_pre_nms[1] / num_images)
        storage.put_scalar("rpn/num_proposals_pre_nms_2", num_pre_nms[2] / num_images)
        storage.put_scalar("rpn/num_proposals_pre_nms_3", num_pre_nms[3] / num_images)
        storage.put_scalar("rpn/num_proposals_post_nms", num_post_nms / num_images)

    # print(results)
    return results


def add_ground_truth_to_proposals(
    gt: Union[List[Instances], List[Boxes]], proposals: List[Instances]
):
    """
    Call `add_ground_truth_to_proposals_single_image` for all images.

    Args:
        gt(Union[List[Instances], List[Boxes]): list of N elements. Element i is a Instances
            representing the ground-truth for image i.
        proposals (list[Instances]): list of N elements. Element i is a Instances
            representing the proposals for image i.

    Returns:
        list[Instances]: list of N Instances. Each is the proposals for the image,
            with field "proposal_boxes" and "objectness_logits".
    """
    assert gt is not None

    if len(proposals) != len(gt):
        raise ValueError("proposals and gt should have the same length as the number of images!")
    if len(proposals) == 0:
        return proposals

    return [
        add_ground_truth_to_proposals_single_image(gt_i, proposals_i)
        for gt_i, proposals_i in zip(gt, proposals)
    ]


def add_ground_truth_to_proposals_single_image(
    gt: Union[Instances, Boxes], proposals: Instances
):
    """
    Augment `proposals` with `gt`.

    Args:
        Same as `add_ground_truth_to_proposals`, but with gt and proposals
        per image.

    Returns:
        Same as `add_ground_truth_to_proposals`, but for only one image.
    """
    if isinstance(gt, Boxes):
        # convert Boxes to Instances
        gt = Instances(proposals.image_size, gt_boxes=gt)

    gt_boxes = gt.gt_boxes
    device = proposals.objectness_logits.device
    # Assign all ground-truth boxes an objectness logit corresponding to
    # P(object) = sigmoid(logit) =~ 1.
    gt_logit_value = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))
    gt_logits = gt_logit_value * torch.ones(len(gt_boxes), device=device)

    # Concatenating gt_boxes with proposals requires them to have the same fields
    gt_proposal = Instances(proposals.image_size, **gt.get_fields())
    gt_proposal.proposal_boxes = gt_boxes
    gt_proposal.objectness_logits = gt_logits

    for key in proposals.get_fields().keys():
        assert gt_proposal.has(
            key
        ), "The attribute '{}' in `proposals` does not exist in `gt`".format(key)

    # NOTE: Instances.cat only use fields from the first item. Extra fields in latter items
    # will be thrown away.
    new_proposals = Instances.cat([proposals, gt_proposal])

    return new_proposals