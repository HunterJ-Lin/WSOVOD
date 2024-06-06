# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import inspect
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import cv2
import detectron2.utils.comm as comm
from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple, batched_nms
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.proposal_generator.proposal_utils import \
    add_ground_truth_to_proposals
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY
from detectron2.data import MetadataCatalog
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.sampling import subsample_labels
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou, PolygonMasks
from detectron2.utils.events import get_event_storage
from wsovod.modeling.roi_heads.fast_rcnn_open_vocabulary import ObjectMiningOutputLayers, InstanceRefinementOutputLayers
from wsovod.modeling.class_heads import OpenVocabularyClassifier
from wsovod.modeling.poolers import ROIPooler
from detectron2.utils.visualizer import Visualizer
from torch import nn

logger = logging.getLogger(__name__)

def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)

def pairwise_iou_wsl(boxes1: Boxes, boxes2: Boxes):
    """
    Given two lists of boxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    boxes1, boxes2 = boxes1.tensor, boxes2.tensor

    width_height_outer = torch.max(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.min(
        boxes1[:, None, :2], boxes2[:, :2]
    )  # [N,M,2]

    width_height_inner = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )  # [N,M,2]

    width_height_outer.abs_()  # [N,M,2]
    outer = width_height_outer.prod(dim=2)  # [N,M]

    sign = width_height_inner.clone()
    sign[sign > 0] = 1
    sign[sign < 0] = 0
    sign = sign.prod(dim=2)
    sign[sign == 0] = -1

    width_height_inner.abs_()  # [N,M,2]
    inter = width_height_inner.prod(dim=2)  # [N,M]

    # handle empty boxes
    iou = torch.where(
        outer > 0, inter / outer * sign, torch.zeros(1, dtype=inter.dtype, device=inter.device)
    )
    return iou


def select_foreground_proposals(
    proposals: List[Instances], bg_label: int
):
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


def select_proposals_with_visible_keypoints(proposals: List[Instances]):
    """
    Args:
        proposals (list[Instances]): a list of N Instances, where N is the
            number of images.

    Returns:
        proposals: only contains proposals with at least one visible keypoint.

    Note that this is still slightly different from Detectron.
    In Detectron, proposals for training keypoint head are re-sampled from
    all the proposals with IOU>threshold & >=1 visible keypoint.

    Here, the proposals are first sampled from all proposals with
    IOU>threshold, then proposals with no visible keypoint are filtered out.
    This strategy seems to make no difference on Detectron and is easier to implement.
    """
    ret = []
    all_num_fg = []
    for proposals_per_image in proposals:
        # If empty/unannotated image (hard negatives), skip filtering for train
        if len(proposals_per_image) == 0:
            ret.append(proposals_per_image)
            continue
        gt_keypoints = proposals_per_image.gt_keypoints.tensor
        # #fg x K x 3
        vis_mask = gt_keypoints[:, :, 2] >= 1
        xs, ys = gt_keypoints[:, :, 0], gt_keypoints[:, :, 1]
        proposal_boxes = proposals_per_image.proposal_boxes.tensor.unsqueeze(dim=1)  # #fg x 1 x 4
        kp_in_box = (
            (xs >= proposal_boxes[:, :, 0])
            & (xs <= proposal_boxes[:, :, 2])
            & (ys >= proposal_boxes[:, :, 1])
            & (ys <= proposal_boxes[:, :, 3])
        )
        selection = (kp_in_box & vis_mask).any(dim=1)
        selection_idxs = nonzero_tuple(selection)[0]
        all_num_fg.append(selection_idxs.numel())
        ret.append(proposals_per_image[selection_idxs])

    storage = get_event_storage()
    storage.put_scalar("keypoint_head/num_fg_samples", np.mean(all_num_fg))
    return ret


@torch.no_grad()
def get_image_level_gt(targets, num_classes):
    if targets is None:
        return None, None, None
    gt_classes_img = [torch.unique(t.gt_classes, sorted=True) for t in targets]
    gt_classes_img_int = [gt.to(torch.int64) for gt in gt_classes_img]
    gt_classes_img_oh = torch.cat(
        [
            torch.zeros(
                (1, num_classes), dtype=torch.float, device=gt_classes_img[0].device
            ).scatter_(1, torch.unsqueeze(gt, dim=0), 1)
            for gt in gt_classes_img_int
        ],
        dim=0,
    )

    return gt_classes_img, gt_classes_img_int, gt_classes_img_oh


class ROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It typically contains logic to

    1. (in training only) match proposals with ground truth and sample them
    2. crop the regions and extract per-region features using proposals
    3. make per-region predictions with different heads

    It can have many variants, implemented as subclasses of this class.
    This base class contains the logic to match/sample proposals.
    But it is not necessary to inherit this class if the sampling logic is not needed.
    """

    @configurable
    def __init__(
        self,
        *,
        num_classes,
        batch_size_per_image,
        positive_fraction,
        proposal_matcher,
        proposal_append_gt=True,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
    ):
        """
        NOTE: this interface is experimental.

        Args:
            num_classes (int): number of foreground classes (i.e. background is not included)
            batch_size_per_image (int): number of proposals to sample for training
            positive_fraction (float): fraction of positive (foreground) proposals
                to sample for training.
            proposal_matcher (Matcher): matcher that matches proposals and ground truth
            proposal_append_gt (bool): whether to include ground truth as proposals as well
        """
        super().__init__()
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.num_classes = num_classes
        self.proposal_matcher = proposal_matcher
        self.proposal_append_gt = proposal_append_gt

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        return {
            "batch_size_per_image": cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
            "positive_fraction": cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "proposal_append_gt": cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT,
            # Matcher to assign box proposals to gt boxes
            "proposal_matcher": Matcher(
                cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
                cfg.MODEL.ROI_HEADS.IOU_LABELS,
                allow_low_quality_matches=False,
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    def _sample_proposals(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
    ):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_idxs = torch.arange(gt_classes.shape[0])
        return sampled_idxs, gt_classes[sampled_idxs]

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_fraction, self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances], suffix=""
    ):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        num_ig_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                proposals_per_image.gt_boxes = targets_per_image.gt_boxes[
                    matched_idxs[sampled_idxs]
                ]
            if has_gt and targets_per_image.has("gt_scores"):
                proposals_per_image.gt_scores = targets_per_image.gt_scores[
                    matched_idxs[sampled_idxs]
                ]
            if has_gt and targets_per_image.has("gt_weights"):
                proposals_per_image.gt_weights = targets_per_image.gt_weights[
                    matched_idxs[sampled_idxs]
                ]
                # alpha = 1 - 1.0 * self.iter / self.max_iter
                # proposals_per_image.gt_weights = (1 - alpha) * proposals_per_image.gt_weights + alpha * proposals_per_image.objectness_logits
                # proposals_per_image.gt_weights = torch.clamp(proposals_per_image.gt_weights, min=1e-6, max=1.0 - 1e-6)
            if has_gt and targets_per_image.has("gt_masks"):
                proposals_per_image.gt_masks = targets_per_image.gt_masks[
                    matched_idxs[sampled_idxs]
                ]

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            num_ig_samples.append((gt_classes == -1).sum().item())
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1] - num_ig_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head_wsl/num_fg_samples" + suffix, np.mean(num_fg_samples))
        storage.put_scalar("roi_head_wsl/num_bg_samples" + suffix, np.mean(num_bg_samples))
        storage.put_scalar("roi_head_wsl/num_ig_samples" + suffix, np.mean(num_ig_samples))

        return proposals_with_gt

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ):
        """
        Args:
            images (ImageList):
            features (dict[str,Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:

                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
                - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.

        Returns:
            list[Instances]: length `N` list of `Instances` containing the
            detected instances. Returned during inference only; may be [] during training.

            dict[str->Tensor]:
            mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        raise NotImplementedError()

@ROI_HEADS_REGISTRY.register()
class WSOVODROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        object_miner: nn.Module,
        sam: nn.Module,
        train_on_pred_boxes: bool = False,
        output_dir: str = None,
        vis_test: bool = False,
        vis_period: int = 0,
        mrrp_on: bool = False,
        mrrp_num_branch: int = 3,
        mrrp_fast: bool = False,
        refine_K: int = 4,
        refine_mist: bool = False,
        refine_reg: List[bool] = [False, False, False, False],
        box_refinery: List[nn.Module] = [None, None, None, None],
        sampling_on: bool = False,
        proposal_matchers: List[Matcher] = [None, None, None, None],
        batch_size_per_images: List[int] = [512, 512, 512, 512],
        positive_sample_fractions: List[float] = [0.25, 0.25, 0.25, 0.25],
        cls_agnostic_bbox_known: bool = False,
        pooler_type: str = "ROIPool",
        rpn_on: bool = False,
        metadata: Dict = None,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            object_miner (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask
                pooler or mask head. None if not using mask head.
            mask_pooler (ROIPooler): pooler to extract region features from image features.
                The mask head will then take region features to make predictions.
                If None, the mask head will directly take the dict of image features
                defined by `mask_in_features`
            mask_head (nn.Module): transform features to make mask predictions
            keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask_*``.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super().__init__(**kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.object_miner = object_miner
        self.sam = sam

        self.train_on_pred_boxes = train_on_pred_boxes

        self.iter = 0
        self.iter_test = 0
        self.epoch_test = 0

        self.output_dir = output_dir
        self.vis_test = vis_test
        self.vis_period = vis_period

        self.mrrp_on = mrrp_on
        self.mrrp_num_branch = mrrp_num_branch
        self.mrrp_fast = mrrp_fast

        self.refine_K = refine_K
        self.refine_mist = refine_mist
        self.refine_reg = refine_reg
        self.box_refinery = box_refinery
        for k in range(self.refine_K):
            self.add_module("box_refinery_{}".format(k), self.box_refinery[k])

        self.sampling_on = sampling_on
        self.proposal_matchers = proposal_matchers
        self.batch_size_per_images = batch_size_per_images
        self.positive_sample_fractions = positive_sample_fractions
        self.cls_agnostic_bbox_known = cls_agnostic_bbox_known
        self.pooler_type = pooler_type
        self.rpn_on = rpn_on

        self.metadata = metadata

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        sam = None
        if cfg.WSOVOD.BBOX_REFINE.ENABLE:
            from segment_anything import sam_model_registry, SamPredictor
            from wsovod.utils.sam_predictor_with_buffer import SamPredictorBuffer
            sam_checkpoint = cfg.WSOVOD.BBOX_REFINE.MODEL_CHECKPOINT
            model_type = cfg.WSOVOD.BBOX_REFINE.MODEL_TYPE
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam = sam.to(device=cfg.MODEL.DEVICE)
        # ret["sam"] = SamPredictor(sam) if sam else None
        ret["sam"] = SamPredictorBuffer(sam) if sam else None
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        mrrp_on = cfg.MODEL.MRRP.MRRP_ON
        mrrp_num_branch = cfg.MODEL.MRRP.NUM_BRANCH
        mrrp_fast = cfg.MODEL.MRRP.TEST_BRANCH_IDX != -1
        if mrrp_on:
            pooler_scales = tuple(
                1.0 / input_shape[k].stride for k in in_features * mrrp_num_branch
            )

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )

        # open_vocabulary_class_head = OpenVocabularyClassifier(cfg, box_head.output_shape)
        # object_miner = ObjectMiningOutputLayers(cfg, box_head.output_shape, open_vocabulary_class_head)
        object_miner = ObjectMiningOutputLayers(cfg, box_head.output_shape, None)

        refine_K = cfg.WSOVOD.INSTANCE_REFINEMENT.REFINE_NUM
        refine_mist = cfg.WSOVOD.INSTANCE_REFINEMENT.REFINE_MIST
        refine_reg = cfg.WSOVOD.INSTANCE_REFINEMENT.REFINE_REG
        box_refinery = []
        for k in range(refine_K):
            open_vocabulary_class_head = OpenVocabularyClassifier(cfg, box_head.output_shape)
            box_refinery_k = InstanceRefinementOutputLayers(cfg, box_head.output_shape, k, open_vocabulary_class_head)
            box_refinery.append(box_refinery_k)

        sampling_on = cfg.WSOVOD.SAMPLING.SAMPLING_ON
        proposal_matchers = [None for _ in range(refine_K)]
        if sampling_on:
            for k in range(refine_K):
                # Matcher to assign box proposals to gt boxes
                proposal_matchers[k] = Matcher(
                    cfg.WSOVOD.SAMPLING.IOU_THRESHOLDS[k],
                    cfg.WSOVOD.SAMPLING.IOU_LABELS[k],
                    allow_low_quality_matches=False,
                )
        batch_size_per_images = cfg.WSOVOD.SAMPLING.BATCH_SIZE_PER_IMAGE
        positive_sample_fractions = cfg.WSOVOD.SAMPLING.POSITIVE_FRACTION

        cls_agnostic_bbox_known = cfg.WSOVOD.CLS_AGNOSTIC_BBOX_KNOWN
        output_dir = cfg.OUTPUT_DIR
        vis_test = cfg.VIS_TEST
        vis_period = cfg.VIS_PERIOD

        rpn_on = False if cfg.MODEL.PROPOSAL_GENERATOR.NAME == "PrecomputedProposals" else True

        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "object_miner": object_miner,
            "output_dir": output_dir,
            "vis_test": vis_test,
            "vis_period": vis_period,
            "mrrp_on": mrrp_on,
            "mrrp_num_branch": mrrp_num_branch,
            "mrrp_fast": mrrp_fast,
            "refine_K": refine_K,
            "refine_mist": refine_mist,
            "refine_reg": refine_reg,
            "box_refinery": box_refinery,
            "sampling_on": sampling_on,
            "proposal_matchers": proposal_matchers,
            "batch_size_per_images": batch_size_per_images,
            "positive_sample_fractions": positive_sample_fractions,
            "cls_agnostic_bbox_known": cls_agnostic_bbox_known,
            "pooler_type": pooler_type,
            "rpn_on": rpn_on,
            "metadata": metadata,
        }

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        data_aware_features = None,
        targets: Optional[List[Instances]] = None,
        classifier = None,
        append_background = True,
        file_names = None,
        loaded_proposals = None,
    ):
        """
        See :class:`ROIHeads.forward`.
        """
        self.gt_classes_img, self.gt_classes_img_int, self.gt_classes_img_oh = get_image_level_gt(
            targets, self.num_classes
        )

        self.images = images
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
            self._vis_proposal(proposals, "train", "_proposals")
        else:
            self._save_proposal_test(proposals, "test", "_proposals")

        del targets

        if self.training:
            losses = self._forward_box(features, proposals, data_aware_features, classifier,append_background,file_names=file_names,loaded_proposals=loaded_proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.

            self.iter = self.iter + 1
            if self.iter_test > 0:
                self.epoch_test = self.epoch_test + 1
            self.iter_test = 0

            return proposals, losses
        else:
            pred_instances, all_scores, all_boxes = self._forward_box(features, proposals, data_aware_features, classifier,append_background)
            
            self.iter_test = self.iter_test + 1

            return pred_instances, {}, all_scores, all_boxes

    def _forward_box(
        self, 
        features: Dict[str, torch.Tensor], 
        proposals: List[Instances], 
        data_aware_features = None, 
        classifier = None,
        append_background = True,
        file_names = None,
        loaded_proposals = None,
    ):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        if self.mrrp_on:
            features = [torch.chunk(f, self.mrrp_num_branch) for f in features]
            features = [ff for f in features for ff in f]

        box_features = self.box_pooler(
            features,
            [x.proposal_boxes for x in proposals],
            level_ids=[torch.div(x.level_ids, 1000, rounding_mode='floor') for x in proposals] if self.mrrp_on else None,
        )

        objectness_logits = torch.cat([x.objectness_logits + 1 for x in proposals], dim=0)
        if self.pooler_type == "ROILoopPool":
            objectness_logits = torch.cat(
                [objectness_logits, objectness_logits, objectness_logits], dim=0
            )

        box_features = box_features * objectness_logits.view(-1, 1, 1, 1)
        if self.training and objectness_logits.numel() > 0:
            storage = get_event_storage()
            storage.put_scalar("proposals/objectness_logits+1 mean", objectness_logits.mean())
            storage.put_scalar("proposals/objectness_logits+1 max", objectness_logits.max())
            storage.put_scalar("proposals/objectness_logits+1 min", objectness_logits.min())

        box_features = self.box_head(box_features)

        if self.pooler_type == "ROILoopPool":
            box_features, box_features_frame, box_features_context = torch.chunk(
                box_features, 3, dim=0
            )
            if data_aware_features is not None:
                box_features = box_features + data_aware_features
                box_features_frame = box_features_frame + data_aware_features
                box_features_context = box_features_context + data_aware_features
            predictions = self.object_miner(
                [box_features, box_features_frame, box_features_context], proposals, context=True
            )
            del box_features_frame
            del box_features_context
        else:
            if data_aware_features is not None:
                box_features += data_aware_features
            predictions = self.object_miner(box_features, proposals)

        if self.training:
            losses = self.object_miner.losses(predictions, proposals, self.gt_classes_img_oh)

            self.pred_class_img_logits = (
                self.object_miner.predict_probs_img(predictions, proposals).clone().detach()
            )

            prev_pred_scores = self.object_miner.predict_probs(predictions, proposals)
            prev_pred_scores = [prev_pred_score.detach() for prev_pred_score in prev_pred_scores]
            prev_pred_boxes = self.object_miner.predict_boxes(predictions, proposals)
            self._vis_prediction(
                prev_pred_boxes,
                prev_pred_scores,
                proposals,
                top_k=100,
                prefix="train",
                suffix="_ObjectMining",
            )
            if self.sam:
                self.sam.reset_buffer()
            for k in range(self.refine_K):
                suffix = "_r" + str(k)
                term_weight = 1
                if self.refine_mist:
                    targets = self.get_pgt_mist(
                        prev_pred_boxes, 
                        prev_pred_scores, 
                        proposals, 
                        sam=self.sam if self.refine_reg[k] else None, 
                        file_names=file_names
                    )
                    self._vis_pgt(targets, "pgt_mist", suffix)
                    if k == 0:
                        term_weight = 1
                else:
                    targets = self.get_pgt_top_k(
                        prev_pred_boxes, 
                        prev_pred_scores, 
                        proposals, 
                        sam=self.sam if self.refine_reg[k] else None, 
                        file_names=file_names
                    )
                    self._vis_pgt(targets, "pgt_top_k", suffix)

                if self.sampling_on:
                    proposals_k = self.label_and_sample_proposals_wsl(
                        k, proposals, targets, suffix=suffix
                    )
                else:
                    proposals_k = self.label_and_sample_proposals(proposals, targets, suffix=suffix)

                predictions_k = self.box_refinery[k](box_features, classifier = classifier, append_background=append_background)

                losses_k = self.box_refinery[k].losses(predictions_k, proposals_k)
                for loss_name in losses_k.keys():
                    losses_k[loss_name] = losses_k[loss_name] * term_weight

                prev_pred_scores = self.box_refinery[k].predict_probs(predictions_k, proposals_k)
                prev_pred_boxes = self.box_refinery[k].predict_boxes(predictions_k, proposals_k)
                prev_pred_scores = [
                    prev_pred_score.detach() for prev_pred_score in prev_pred_scores
                ]
                prev_pred_boxes = [prev_pred_box.detach() for prev_pred_box in prev_pred_boxes]

                self._vis_prediction(
                    prev_pred_boxes,
                    prev_pred_scores,
                    proposals,
                    top_k=100,
                    prefix="train",
                    suffix=suffix,
                )

                losses.update(losses_k)

            if self.rpn_on:
                if loaded_proposals and False:
                    keeps = [
                        batched_nms(
                            prop.proposal_boxes.tensor, 
                            prop.objectness_logits, 
                            torch.zeros_like(prop.objectness_logits), 
                            0.2
                        )
                        for prop in loaded_proposals
                    ]
                    targets = [
                        Instances(
                            prop.image_size,
                            gt_boxes=prop.proposal_boxes[keep],
                            gt_scores=prop.objectness_logits[keep],
                            gt_classes=torch.zeros(prop.objectness_logits[keep].shape,dtype=torch.int),
                        )
                        for prop,keep in zip(loaded_proposals,keeps)
                    ]
                    self._vis_pgt(targets, "loaded_proposals", "_rpn")
                else:
                    # targets = self.get_pgt_mist(
                    #     prev_pred_boxes, 
                    #     prev_pred_scores, 
                    #     proposals, 
                    #     top_pro=0.05, 
                    #     # sam=self.sam, 
                    #     # file_names=file_names
                    # )
                    # self._vis_pgt(targets, "get_pgt_mist", "_rpn")
                    targets = self.get_pgt_top_k(
                        prev_pred_boxes, 
                        prev_pred_scores, 
                        proposals, 
                        top_k=1, 
                        sam=self.sam, 
                        file_names=file_names
                    )
                    self._vis_pgt(targets, "pgt_top_k", "_rpn")
                self.proposal_targets = targets

            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.object_miner.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        
        else:
            if self.refine_K > 0:
                predictions_K = []
                for k in range(self.refine_K):
                    predictions_k = self.box_refinery[k](box_features, classifier, append_background)
                    predictions_K.append(predictions_k)
                pred_instances, _, all_scores, all_boxes = self.box_refinery[-1].inference(
                    predictions_K, proposals
                )
            else:
                predictions = self.object_miner(box_features, proposals, context=True)
                pred_instances, _, all_scores, all_boxes = self.box_predictor.inference(
                    predictions, proposals
                )
            return pred_instances, all_scores, all_boxes

    @torch.no_grad()
    def get_pgt_mist(
        self, 
        prev_pred_boxes, 
        prev_pred_scores, 
        proposals, 
        top_pro=0.15,
        sam=None,
        file_names=None
    ):
        pgt_scores, pgt_boxes, pgt_classes, pgt_weights = self.get_pgt_top_k(
            prev_pred_boxes,
            prev_pred_scores,
            proposals,
            top_k=top_pro,
            thres=0.05,
            # thres=0.0,
            need_instance=False,
            need_weight=True,
        )

        # NMS
        pgt_idxs = [torch.zeros_like(pgt_class) for pgt_class in pgt_classes]
        keeps = [
            batched_nms(pgt_box, pgt_score, pgt_class, 0.2)
            for pgt_box, pgt_score, pgt_class in zip(pgt_boxes, pgt_scores, pgt_idxs)
        ]
        pgt_scores = [pgt_score[keep] for pgt_score, keep in zip(pgt_scores, keeps)]
        pgt_boxes = [pgt_box[keep] for pgt_box, keep in zip(pgt_boxes, keeps)]
        pgt_classes = [pgt_class[keep] for pgt_class, keep in zip(pgt_classes, keeps)]
        pgt_weights = [pgt_weight[keep] for pgt_weight, keep in zip(pgt_weights, keeps)]

        # sam refine
        pgt_boxes_old = [None for _ in range(len(self.images))]
        polygons_masks_per_image = [None for _ in range(len(self.images))]
        if sam:
            pgt_boxes_old = [Boxes(pgt_box.clone()) for pgt_box in pgt_boxes]
            bitmasks_per_image = []
            for i,pgt_box in enumerate(pgt_boxes):
                center_x = (pgt_box[:,0] + pgt_box[:,2]) / 2
                center_y = (pgt_box[:,1] + pgt_box[:,3]) / 2
                width = pgt_box[:,2] - pgt_box[:,0]
                height = pgt_box[:,3] - pgt_box[:,1]
                width *= 1.1
                height *= 1.1
                # width *= 1.0
                # height *= 1.0
                new_x1 = center_x - width / 2
                new_y1 = center_y - height / 2
                new_x2 = center_x + width / 2
                new_y2 = center_y + height / 2
                x1 = new_x1.clamp(min=0, max=self.images[i].shape[-1])
                y1 = new_y1.clamp(min=0, max=self.images[i].shape[-2])
                x2 = new_x2.clamp(min=0, max=self.images[i].shape[-1])
                y2 = new_y2.clamp(min=0, max=self.images[i].shape[-2])
                input_boxes = torch.stack((x1, y1, x2, y2), dim=-1)
                transformed_boxes = sam.transform.apply_boxes_torch(input_boxes, self.images[i].shape[-2:])
                sam.set_image(
                    self.images[i].cpu().clone().numpy().astype(np.uint8).transpose(1,2,0),
                    image_format="BGR",
                    file_name=file_names[i],
                )
                bitmasks, mask_scores, _ = sam.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                    file_name=file_names[i],
                )
                bitmasks = bitmasks.squeeze(1)
                mask_scores = mask_scores.squeeze(1)
                bitmasks_per_image.append(bitmasks)

            def mask_to_polygons(mask):
                # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
                # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
                # Internal contours (holes) are placed in hierarchy-2.
                # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
                mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
                res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                hierarchy = res[-1]
                if hierarchy is None:  # empty mask
                    return [], False
                has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
                res = res[-2]
                res = [x.flatten() for x in res]
                # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
                # We add 0.5 to turn them into real-value coordinate space. A better solution
                # would be to first +0.5 and then dilate the returned polygon by 0.5.
                res = [x + 0.5 for x in res if len(x) >= 6]
                return res, has_holes

            polygons_masks_per_image = [
                    [mask_to_polygons(bitmask)[0] for bitmask in bitmasks.cpu().numpy()]
                    for bitmasks in bitmasks_per_image
            ]
            polygons_masks_per_image = [PolygonMasks(polygons_masks) for polygons_masks in polygons_masks_per_image]
            pgt_boxes = [polygons_masks.get_bounding_boxes().tensor.to(sam.device) for polygons_masks in polygons_masks_per_image]


        pgt_boxes = [Boxes(pgt_box) for pgt_box in pgt_boxes]

        if sam:
            targets = [
                Instances(
                    proposals[i].image_size,
                    gt_boxes=pgt_box,
                    ori_pgt_boxes=ori_pgt_box,
                    gt_masks=pgt_masks,
                    gt_classes=pgt_class,
                    gt_scores=pgt_score,
                    gt_weights=pgt_weight,
                )
                for i, (pgt_box, ori_pgt_box, pgt_masks, pgt_class, pgt_score, pgt_weight) in enumerate(
                    zip(pgt_boxes, pgt_boxes_old, polygons_masks_per_image, pgt_classes, pgt_scores, pgt_weights)
                )
            ]
        else:
            targets = [
                Instances(
                    proposals[i].image_size,
                    gt_boxes=pgt_box,
                    gt_classes=pgt_class,
                    gt_scores=pgt_score,
                    gt_weights=pgt_weight,
                )
                for i, (pgt_box, pgt_class, pgt_score, pgt_weight) in enumerate(
                    zip(pgt_boxes, pgt_classes, pgt_scores, pgt_scores)
                )
            ]
            
        return targets

    @torch.no_grad()
    def get_pgt_top_k(
        self,
        prev_pred_boxes,
        prev_pred_scores,
        proposals,
        top_k=1,
        thres=0,
        need_instance=True,
        need_weight=True,
        sam=None,
        file_names=None,
    ):
        assert isinstance(prev_pred_boxes, tuple) or isinstance(prev_pred_boxes, list)

        assert isinstance(prev_pred_boxes[0], torch.Tensor)
        num_preds = [prev_pred_box.size(0) for prev_pred_box in prev_pred_boxes]
        if prev_pred_boxes[0].size(1) == 4:
            prev_pred_boxes = [
                prev_pred_box.unsqueeze(1).expand(num_pred, self.num_classes, 4)
                for num_pred, prev_pred_box in zip(num_preds, prev_pred_boxes)
            ]
        else:
            assert (prev_pred_boxes[0].size(1) == self.num_classes * 4) or (prev_pred_boxes[0].size(1) == self.num_classes and prev_pred_boxes[0].size(2) == 4)
            prev_pred_boxes = [
                prev_pred_box.view(-1, self.num_classes, 4) for prev_pred_box in prev_pred_boxes
            ]

        if isinstance(prev_pred_scores, torch.Tensor):
            num_preds_per_image = [len(p) for p in proposals]
            prev_pred_scores = prev_pred_scores.split(num_preds_per_image, dim=0)
        else:
            assert isinstance(prev_pred_scores, tuple) or isinstance(
                prev_pred_scores, list
            ), prev_pred_scores
            assert isinstance(prev_pred_scores[0], torch.Tensor), prev_pred_scores[0]

        prev_pred_scores = [
            torch.index_select(prev_pred_score, 1, gt_int)
            for prev_pred_score, gt_int in zip(prev_pred_scores, self.gt_classes_img_int)
        ]
        prev_pred_boxes = [
            torch.index_select(prev_pred_box, 1, gt_int)
            for prev_pred_box, gt_int in zip(prev_pred_boxes, self.gt_classes_img_int)
        ]


        # filter small pgt
        def get_area(box):
            return (box[:,:,2]-box[:,:,0])*(box[:,:,3]-box[:,:,1])
        
        prev_pred_boxes_keep = [
            get_area(box)>20
            for box in prev_pred_boxes
        ]

        prev_pred_boxes = [
            boxes.masked_select(
                torch.unsqueeze(mask, 2).expand(-1, gt_int.numel(), 4)
            ).view(-1,gt_int.numel(), 4)
            for boxes, mask, gt_int in zip(
                prev_pred_boxes, prev_pred_boxes_keep, self.gt_classes_img_int
            )
        ]
        prev_pred_scores = [
            scores.masked_select(mask).view(-1, gt_int.numel())
            for scores, mask, gt_int in zip(
                prev_pred_scores, prev_pred_boxes_keep, self.gt_classes_img_int
            )
        ]


        # get top k
        num_preds = [prev_pred_score.size(0) for prev_pred_score in prev_pred_scores]
        if top_k >= 1:
            top_ks = [min(num_pred, int(top_k)) for num_pred in num_preds]
        elif top_k < 1 and top_k > 0:
            top_ks = [max(int(num_pred * top_k), 1) for num_pred in num_preds]
        else:
            top_ks = [min(num_pred, 1) for num_pred in num_preds]
        pgt_scores_idxs = [
            torch.topk(prev_pred_score, top_k, dim=0)
            for prev_pred_score, top_k in zip(prev_pred_scores, top_ks)
        ]
        pgt_scores = [item[0] for item in pgt_scores_idxs]
        pgt_idxs = [item[1] for item in pgt_scores_idxs]
        pgt_idxs = [
            torch.unsqueeze(pgt_idx, 2).expand(top_k, gt_int.numel(), 4)
            for pgt_idx, top_k, gt_int in zip(pgt_idxs, top_ks, self.gt_classes_img_int)
        ]
        pgt_boxes = [
            torch.gather(prev_pred_box, 0, pgt_idx)
            for prev_pred_box, pgt_idx in zip(prev_pred_boxes, pgt_idxs)
        ]
        pgt_classes = [
            torch.unsqueeze(gt_int, 0).expand(top_k, gt_int.numel())
            for gt_int, top_k in zip(self.gt_classes_img_int, top_ks)
        ]
        if need_weight:
            pgt_weights = [
                torch.index_select(pred_logits, 1, gt_int).expand(top_k, gt_int.numel())
                for pred_logits, gt_int, top_k in zip(
                    self.pred_class_img_logits.split(1, dim=0), self.gt_classes_img_int, top_ks
                )
            ]

        if thres > 0:
            # get large scores
            masks = [pgt_score.ge(thres) for pgt_score in pgt_scores]
            masks = [
                torch.cat([torch.full_like(mask[0:1, :], True), mask[1:, :]], dim=0)
                for mask in masks
            ]
            pgt_scores = [
                torch.masked_select(pgt_score, mask) for pgt_score, mask in zip(pgt_scores, masks)
            ]
            pgt_boxes = [
                torch.masked_select(
                    pgt_box, torch.unsqueeze(mask, 2).expand(top_k, gt_int.numel(), 4)
                )
                for pgt_box, mask, top_k, gt_int in zip(
                    pgt_boxes, masks, top_ks, self.gt_classes_img_int
                )
            ]
            pgt_classes = [
                torch.masked_select(pgt_class, mask) for pgt_class, mask in zip(pgt_classes, masks)
            ]
            if need_weight:
                pgt_weights = [
                    torch.masked_select(pgt_weight, mask)
                    for pgt_weight, mask in zip(pgt_weights, masks)
                ]

        pgt_scores = [pgt_score.reshape(-1) for pgt_score in pgt_scores]
        pgt_boxes = [pgt_box.reshape(-1, 4) for pgt_box in pgt_boxes]
        pgt_classes = [pgt_class.reshape(-1) for pgt_class in pgt_classes]
        if need_weight:
            pgt_weights = [pgt_weight.reshape(-1) for pgt_weight in pgt_weights]

            pgt_weights = [
                pgt_weight
                if pgt_weight.numel() > 0
                else torch.tensor([1], dtype=pgt_weight.dtype, device=pgt_weight.device)
                for pgt_weight in pgt_weights
            ]

        pgt_scores = [
            pgt_score
            if pgt_score.numel() > 0
            else torch.tensor([1], dtype=pgt_score.dtype, device=pgt_score.device)
            for pgt_score in pgt_scores
        ]
        pgt_boxes = [
            pgt_box
            if pgt_box.numel() > 0
            else torch.tensor(
                [[-10000, -10000, 10000, 10000]], dtype=pgt_box.dtype, device=pgt_box.device
            )
            for pgt_box in pgt_boxes
        ]
        pgt_classes = [
            pgt_class
            if pgt_class.numel() > 0
            else torch.tensor([0], dtype=pgt_class.dtype, device=pgt_class.device)
            for pgt_class in pgt_classes
        ]

        if not need_instance:
            if need_weight:
                return pgt_scores, pgt_boxes, pgt_classes, pgt_weights
            else:
                return pgt_scores, pgt_boxes, pgt_classes

        # sam refine
        pgt_boxes_old = [None for _ in range(len(self.images))]
        polygons_masks_per_image = [None for _ in range(len(self.images))]
        if sam:
            pgt_boxes_old = [Boxes(pgt_box.clone()) for pgt_box in pgt_boxes]
            bitmasks_per_image = []
            for i,pgt_box in enumerate(pgt_boxes):
                center_x = (pgt_box[:,0] + pgt_box[:,2]) / 2
                center_y = (pgt_box[:,1] + pgt_box[:,3]) / 2
                width = pgt_box[:,2] - pgt_box[:,0]
                height = pgt_box[:,3] - pgt_box[:,1]
                width *= 1.1
                height *= 1.1
                # width *= 1.0
                # height *= 1.0
                new_x1 = center_x - width / 2
                new_y1 = center_y - height / 2
                new_x2 = center_x + width / 2
                new_y2 = center_y + height / 2
                x1 = new_x1.clamp(min=0, max=self.images[i].shape[-1])
                y1 = new_y1.clamp(min=0, max=self.images[i].shape[-2])
                x2 = new_x2.clamp(min=0, max=self.images[i].shape[-1])
                y2 = new_y2.clamp(min=0, max=self.images[i].shape[-2])
                input_boxes = torch.stack((x1, y1, x2, y2), dim=-1)
                transformed_boxes = sam.transform.apply_boxes_torch(input_boxes, self.images[i].shape[-2:])
                sam.set_image(
                    self.images[i].cpu().clone().numpy().astype(np.uint8).transpose(1,2,0),
                    image_format="BGR",
                    file_name=file_names[i],
                )
                bitmasks, mask_scores, _ = sam.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                    file_name=file_names[i],
                )
                bitmasks = bitmasks.squeeze(1)
                mask_scores = mask_scores.squeeze(1)
                bitmasks_per_image.append(bitmasks)

            def mask_to_polygons(mask):
                # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
                # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
                # Internal contours (holes) are placed in hierarchy-2.
                # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
                mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
                res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                hierarchy = res[-1]
                if hierarchy is None:  # empty mask
                    return [], False
                has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
                res = res[-2]
                res = [x.flatten() for x in res]
                # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
                # We add 0.5 to turn them into real-value coordinate space. A better solution
                # would be to first +0.5 and then dilate the returned polygon by 0.5.
                res = [x + 0.5 for x in res if len(x) >= 6]
                return res, has_holes

            polygons_masks_per_image = [
                [mask_to_polygons(bitmask)[0] for bitmask in bitmasks.cpu().numpy()]
                for bitmasks in bitmasks_per_image
            ]
            polygons_masks_per_image = [PolygonMasks(polygons_masks) for polygons_masks in polygons_masks_per_image]
            for i,polygons_masks in enumerate(polygons_masks_per_image):
                pgt_box = polygons_masks.get_bounding_boxes().tensor.to(sam.device)
                inf_indices = torch.any(pgt_box == float('inf'), dim=1).nonzero(as_tuple=True)[0]
                pgt_box[inf_indices] = pgt_boxes[i][inf_indices]
                pgt_boxes[i] = pgt_box

        pgt_boxes = [Boxes(pgt_box) for pgt_box in pgt_boxes]
        if sam:
            if need_weight:
                targets = [
                    Instances(
                        proposals[i].image_size,
                        gt_boxes=pgt_box,
                        ori_pgt_boxes=ori_pgt_box,
                        gt_masks=pgt_masks,
                        gt_classes=pgt_class,
                        gt_scores=pgt_score,
                        gt_weights=pgt_weight,
                    )
                    for i, (pgt_box, ori_pgt_box, pgt_masks, pgt_class, pgt_score, pgt_weight) in enumerate(
                        zip(pgt_boxes, pgt_boxes_old, polygons_masks_per_image, pgt_classes, pgt_scores, pgt_weights)
                    )
                ]
            else:
                targets = [
                    Instances(
                        proposals[i].image_size,
                        gt_boxes=pgt_box,
                        pgt_boxes=ori_pgt_box,
                        gt_classes=pgt_class,
                        gt_scores=pgt_score,
                    )
                    for i, (pgt_box, ori_pgt_box, pgt_class, pgt_score) in enumerate(
                        zip(pgt_boxes, pgt_boxes_old, pgt_classes, pgt_scores)
                    )
                ]
        else:
            if need_weight:
                targets = [
                    Instances(
                        proposals[i].image_size,
                        gt_boxes=pgt_box,
                        gt_classes=pgt_class,
                        gt_scores=pgt_score,
                        gt_weights=pgt_weight,
                    )
                    for i, (pgt_box, pgt_class, pgt_score, pgt_weight) in enumerate(
                        zip(pgt_boxes, pgt_classes, pgt_scores, pgt_weights)
                    )
                ]
            else:
                targets = [
                    Instances(
                        proposals[i].image_size,
                        gt_boxes=pgt_box,
                        gt_classes=pgt_class,
                        gt_scores=pgt_score,
                    )
                    for i, (pgt_box, pgt_class, pgt_score) in enumerate(
                        zip(pgt_boxes, pgt_classes, pgt_scores)
                    )
                ]            

        return targets

    @torch.no_grad()
    def _vis_pgt(self, targets, prefix, suffix, thickness=4):
        if self.vis_period <= 0 or self.iter % self.vis_period > 0:
            return
        storage = get_event_storage()

        targets = copy.deepcopy(targets)

        output_dir = os.path.join(self.output_dir, prefix)
        if self.iter == 0:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        for b, target in enumerate(targets):
            img = self.images.tensor[b, ...].clone().detach() * self.pixel_std + self.pixel_mean
            img = img.cpu().numpy().transpose((1, 2, 0))
            img = img.astype(np.uint8)
            h, w = img.shape[:2]
            img = img.copy()

            # if target.has("gt_scores"):
            #     target.scores = target.gt_scores.clone().detach().cpu()
            # if target.has("gt_boxes"):
            #     target.pred_boxes = target.gt_boxes.tensor.clone().detach().cpu()
            # if target.has("gt_classes"):
            #     target.pred_classes = target.gt_classes.clone().detach().cpu()

            vis = Visualizer(img, self.metadata)
            # vis._default_font_size = max(
            #     np.sqrt(h * w) // 40, 20
            # )
            # vis_pred = vis.draw_instance_predictions(target).get_image()

            def _create_text_labels(classes, scores, class_names, is_crowd=None):
                """
                Args:
                    classes (list[int] or None):
                    scores (list[float] or None):
                    class_names (list[str] or None):
                    is_crowd (list[bool] or None):
                Returns:
                    list[str] or None
                """
                labels = None
                if classes is not None:
                    if class_names is not None and len(class_names) > 0:
                        labels = [class_names[i] for i in classes]
                    else:
                        labels = [str(i) for i in classes]
                if scores is not None:
                    if labels is None:
                        labels = ["{:.0f}%".format(s * 100) for s in scores]
                    else:
                        labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
                if labels is not None and is_crowd is not None:
                    labels = [l + ("|crowd" if crowd else "") for l, crowd in zip(labels, is_crowd)]
                return labels
            
            vis_pred = vis.overlay_instances(
                boxes=target.gt_boxes.tensor.clone().detach().cpu(),
                labels=_create_text_labels(target.gt_classes,target.gt_scores,self.metadata.thing_classes) if target.has('gt_classes') else None,
                masks=target.gt_masks if target.has('gt_masks') else None,
            ).get_image()
            if target.has('ori_pgt_boxes'):
                v_gt_ori = Visualizer(img, self.metadata)
                v_gt_ori = v_gt_ori.overlay_instances(
                    boxes=target.ori_pgt_boxes.tensor.clone().detach().cpu(),
                    labels=_create_text_labels(target.gt_classes,target.gt_scores,self.metadata.thing_classes)
                )
                vis_pred = np.concatenate((v_gt_ori.get_image(), vis_pred), axis=1)

            device_index = comm.get_rank()
            save_name = (
                "i" + str(self.iter) + "_g" + str(device_index) + "_b" + str(b) + suffix + ".png"
            )

            save_path = os.path.join(output_dir, save_name)
            cv2.imwrite(save_path, vis_pred)

            vis_pred = vis_pred.transpose(2, 0, 1)
            vis_name = prefix + "_g" + str(device_index) + "_b" + str(b) + suffix
            storage.put_image(vis_name, vis_pred)

    @torch.no_grad()
    def _vis_prediction(
        self,
        prev_pred_boxes,
        prev_pred_scores,
        proposals,
        top_k=1,
        thres=0.01,
        thickness=4,
        prefix="",
        suffix="",
    ):
        if self.vis_period <= 0 or self.iter % self.vis_period > 0:
            return

        targets = self.get_pgt_top_k(
            prev_pred_boxes,
            prev_pred_scores,
            proposals,
            top_k=top_k,
            thres=thres,
            need_weight=False,
        )

        self._vis_pgt(targets, prefix, suffix, thickness)

    @torch.no_grad()
    def _vis_proposal(self, proposals, prefix, suffix):
        if self.vis_period <= 0 or self.iter % self.vis_period > 0:
            return

        prev_pred_boxes = [p.proposal_boxes for p in proposals]
        num_preds = [len(prev_pred_box) for prev_pred_box in proposals]
        prev_pred_boxes = [
            prev_pred_box.tensor.unsqueeze(1).expand(num_pred, self.num_classes, 4)
            for num_pred, prev_pred_box in zip(num_preds, prev_pred_boxes)
        ]

        prev_pred_scores = [p.objectness_logits for p in proposals]
        prev_pred_scores = [
            prev_pred_score.unsqueeze(1).expand(num_pred, self.num_classes + 1)
            for num_pred, prev_pred_score in zip(num_preds, prev_pred_scores)
        ]

        self._vis_prediction(
            prev_pred_boxes,
            prev_pred_scores,
            proposals,
            top_k=2048,
            thres=-9999,
            thickness=1,
            prefix=prefix,
            suffix=suffix,
        )

    @torch.no_grad()
    def _save_proposal_test(self, proposals, prefix, suffix):
        if self.training or not self.vis_test:
            return

        output_dir = os.path.join(self.output_dir, prefix)
        if self.iter_test == 0:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        for b, p in enumerate(proposals):
            box = p.proposal_boxes.tensor.clone().detach().cpu().numpy()
            logit = p.objectness_logits.clone().detach().cpu().numpy()
            level_ids = p.level_ids.clone().detach().cpu().numpy()

            gpu_id = p.objectness_logits.device.index
            id_str = "i" + str(self.iter_test) + "_g" + str(gpu_id) + "_b" + str(b)

            save_path = os.path.join(output_dir, id_str + "_box" + suffix + ".npy")
            np.save(save_path, box)

            save_path = os.path.join(output_dir, id_str + "_logit" + suffix + ".npy")
            np.save(save_path, logit)

            save_path = os.path.join(output_dir, id_str + "_level" + suffix + ".npy")
            np.save(save_path, level_ids)

    @torch.no_grad()
    def _vis_box(
        self,
        prev_pred_boxes,
        prev_pred_scores,
        proposals,
        top_k=1,
        thres=0.01,
        thickness=4,
        prefix="",
        suffix="",
    ):
        if self.vis_period <= 0 or self.iter % self.vis_period > 0:
            return
        storage = get_event_storage()

        pgt_scores, pgt_boxes, pgt_classes = self.get_pgt_top_k(
            prev_pred_boxes,
            prev_pred_scores,
            proposals,
            top_k=top_k,
            thres=thres,
            need_instance=False,
            need_weight=False,
            suffix="",
        )

        output_dir = os.path.join(self.output_dir, prefix)
        if self.iter == 0:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        for b, pgt_box in enumerate(pgt_boxes):
            img = self.images.tensor[b, ...].clone().detach() * self.pixel_std + self.pixel_mean
            img = img.cpu().numpy().transpose((1, 2, 0))
            img = img.astype(np.uint8)
            h, w = img.shape[:2]
            img_pgt = img.copy()

            device_index = pgt_box.device.index
            save_name = (
                "i" + str(self.iter) + "_g" + str(device_index) + "_b" + str(b) + suffix + ".png"
            )
            pgt_box = pgt_box.cpu().numpy()
            for i in range(pgt_box.shape[0]):
                x0, y0, x1, y1 = pgt_box[i, :]
                x0 = int(max(x0, 0))
                y0 = int(max(y0, 0))
                x1 = int(min(x1, w))
                y1 = int(min(y1, h))
                cv2.rectangle(img_pgt, (x0, y0), (x1, y1), (0, 0, 255), thickness)

            save_path = os.path.join(output_dir, save_name)
            cv2.imwrite(save_path, img_pgt)

            img_pgt = img_pgt.transpose(2, 0, 1)
            vis_name = prefix + "_g" + str(device_index) + "_b" + str(b) + suffix
            storage.put_image(vis_name, img_pgt)

    def _sample_proposals_wsl(
        self, k, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
    ):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes,
            self.batch_size_per_images[k],
            self.positive_sample_fractions[k],
            self.num_classes,
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)

        gt_classes_sp = torch.full_like(gt_classes, -1)
        gt_classes_sp[sampled_idxs] = gt_classes[sampled_idxs]

        sampled_idxs = torch.arange(gt_classes.shape[0])
        return sampled_idxs, gt_classes_sp[sampled_idxs]

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_fraction, self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances], suffix=""
    ):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        num_ig_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt and not self.cls_agnostic_bbox_known:
                proposals_per_image.gt_boxes = targets_per_image.gt_boxes[
                    matched_idxs[sampled_idxs]
                ]
            if has_gt and targets_per_image.has("gt_scores"):
                proposals_per_image.gt_scores = targets_per_image.gt_scores[
                    matched_idxs[sampled_idxs]
                ]
            if has_gt and targets_per_image.has("gt_weights"):
                proposals_per_image.gt_weights = targets_per_image.gt_weights[
                    matched_idxs[sampled_idxs]
                ]

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            num_ig_samples.append((gt_classes == -1).sum().item())
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1] - num_ig_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples" + suffix, np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples" + suffix, np.mean(num_bg_samples))
        storage.put_scalar("roi_head/num_ig_samples" + suffix, np.mean(num_ig_samples))

        return proposals_with_gt

    @torch.no_grad()
    def label_and_sample_proposals_wsl(
        self, k: int, proposals: List[Instances], targets: List[Instances], suffix=""
    ):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        num_ig_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matchers[k](match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals_wsl(
                k, matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt and not self.cls_agnostic_bbox_known:
                proposals_per_image.gt_boxes = targets_per_image.gt_boxes[
                    matched_idxs[sampled_idxs]
                ]
            if has_gt and targets_per_image.has("gt_scores"):
                proposals_per_image.gt_scores = targets_per_image.gt_scores[
                    matched_idxs[sampled_idxs]
                ]
            if has_gt and targets_per_image.has("gt_weights"):
                proposals_per_image.gt_weights = targets_per_image.gt_weights[
                    matched_idxs[sampled_idxs]
                ]
            if has_gt and targets_per_image.has("gt_masks"):
                proposals_per_image.gt_masks = targets_per_image.gt_masks[
                    matched_idxs[sampled_idxs]
                ]

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            num_ig_samples.append((gt_classes == -1).sum().item())
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1] - num_ig_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples" + suffix, np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples" + suffix, np.mean(num_bg_samples))
        storage.put_scalar("roi_head/num_ig_samples" + suffix, np.mean(num_ig_samples))

        return proposals_with_gt

    def get_features(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        data_aware_features = None,
    ):
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])

        objectness_logits = torch.cat([x.objectness_logits + 1 for x in proposals], dim=0)
        if self.pooler_type == "ROILoopPool":
            objectness_logits = torch.cat(
                [objectness_logits, objectness_logits, objectness_logits], dim=0
            )
        box_features = box_features * objectness_logits.view(-1, 1, 1, 1)
        box_features = self.box_head(box_features)
        
        if self.pooler_type == "ROILoopPool":
            box_features, box_features_frame, box_features_context = torch.chunk(
                box_features, 3, dim=0
            )
            if data_aware_features is not None:
                box_features = box_features + data_aware_features
                box_features_frame = box_features_frame + data_aware_features
                box_features_context = box_features_context + data_aware_features
            del box_features_frame
            del box_features_context
        else:
            if data_aware_features is not None:
                box_features += data_aware_features
        return box_features


@ROI_HEADS_REGISTRY.register()
class WSOVODMixedDatasetsROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        object_miners: nn.ModuleList,
        num_classes_list: List=[],
        sam: nn.Module,
        train_on_pred_boxes: bool = False,
        output_dir: str = None,
        vis_test: bool = False,
        vis_period: int = 0,
        mrrp_on: bool = False,
        mrrp_num_branch: int = 3,
        mrrp_fast: bool = False,
        refine_K: int = 4,
        refine_mist: bool = False,
        refine_reg: List[bool] = [False, False, False, False],
        box_refinery: List[nn.Module] = [None, None, None, None],
        sampling_on: bool = False,
        proposal_matchers: List[Matcher] = [None, None, None, None],
        batch_size_per_images: List[int] = [512, 512, 512, 512],
        positive_sample_fractions: List[float] = [0.25, 0.25, 0.25, 0.25],
        cls_agnostic_bbox_known: bool = False,
        pooler_type: str = "ROIPool",
        rpn_on: bool = False,
        metadatas: List[Dict] = [],
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask
                pooler or mask head. None if not using mask head.
            mask_pooler (ROIPooler): pooler to extract region features from image features.
                The mask head will then take region features to make predictions.
                If None, the mask head will directly take the dict of image features
                defined by `mask_in_features`
            mask_head (nn.Module): transform features to make mask predictions
            keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask_*``.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super().__init__(**kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.object_miners = object_miners
        self.num_classes_list = num_classes_list
        self.sam = sam
        
        self.train_on_pred_boxes = train_on_pred_boxes

        self.iter = 0
        self.iter_test = 0
        self.epoch_test = 0

        self.output_dir = output_dir
        self.vis_test = vis_test
        self.vis_period = vis_period

        self.mrrp_on = mrrp_on
        self.mrrp_num_branch = mrrp_num_branch
        self.mrrp_fast = mrrp_fast

        self.refine_K = refine_K
        self.refine_mist = refine_mist
        self.refine_reg = refine_reg
        self.box_refinery = box_refinery
        for k in range(self.refine_K):
            self.add_module("box_refinery_{}".format(k), self.box_refinery[k])

        self.sampling_on = sampling_on
        self.proposal_matchers = proposal_matchers
        self.batch_size_per_images = batch_size_per_images
        self.positive_sample_fractions = positive_sample_fractions
        self.cls_agnostic_bbox_known = cls_agnostic_bbox_known
        self.pooler_type = pooler_type
        self.rpn_on = rpn_on

        self.metadatas = metadatas

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        sam = None
        if cfg.WSOVOD.BBOX_REFINE.ENABLE:
            from segment_anything import sam_model_registry, SamPredictor
            from wsovod.utils.sam_predictor_with_buffer import SamPredictorBuffer
            sam_checkpoint = cfg.WSOVOD.BBOX_REFINE.MODEL_CHECKPOINT
            model_type = cfg.WSOVOD.BBOX_REFINE.MODEL_TYPE
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam = sam.to(device=cfg.MODEL.DEVICE)
        # ret["sam"] = SamPredictor(sam) if sam else None
        ret["sam"] = SamPredictorBuffer(sam) if sam else None
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        mrrp_on = cfg.MODEL.MRRP.MRRP_ON
        mrrp_num_branch = cfg.MODEL.MRRP.NUM_BRANCH
        mrrp_fast = cfg.MODEL.MRRP.TEST_BRANCH_IDX != -1
        if mrrp_on:
            pooler_scales = tuple(
                1.0 / input_shape[k].stride for k in in_features * mrrp_num_branch
            )

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )

        # open_vocabulary_class_head = OpenVocabularyClassifier(cfg, box_head.output_shape)
        miner_dict = {}
        keys_map = {}
        for name in cfg.DATASETS.MIXED_DATASETS.NAMES:
            if 'voc' in name:
                keys_map[name] = 'voc'
            if 'coco' in name:
                keys_map[name] = 'coco'
            if 'lvis' in name:
                keys_map[name] = 'lvis'
            
        for name,num_classes in zip(cfg.DATASETS.MIXED_DATASETS.NAMES,cfg.DATASETS.MIXED_DATASETS.NUM_CLASSES):
            k = keys_map[name]
            if k not in miner_dict.keys():
                object_miner = ObjectMiningOutputLayers(cfg, box_head.output_shape, class_head=None, num_classes=num_classes)
                miner_dict[k] = object_miner

        object_miners = nn.ModuleList([
            miner_dict[keys_map[name]]
            for name in cfg.DATASETS.MIXED_DATASETS.NAMES
        ])

        refine_K = cfg.WSOVOD.INSTANCE_REFINEMENT.REFINE_NUM
        refine_mist = cfg.WSOVOD.INSTANCE_REFINEMENT.REFINE_MIST
        refine_reg = cfg.WSOVOD.INSTANCE_REFINEMENT.REFINE_REG
        box_refinery = []
        assert refine_K>0
        for k in range(refine_K):
            open_vocabulary_class_head = OpenVocabularyClassifier(cfg, box_head.output_shape)
            box_refinery_k = InstanceRefinementOutputLayers(cfg, box_head.output_shape, k, open_vocabulary_class_head)
            box_refinery.append(box_refinery_k)

        sampling_on = cfg.WSOVOD.SAMPLING.SAMPLING_ON
        proposal_matchers = [None for _ in range(refine_K)]
        if sampling_on:
            for k in range(refine_K):
                # Matcher to assign box proposals to gt boxes
                proposal_matchers[k] = Matcher(
                    cfg.WSOVOD.SAMPLING.IOU_THRESHOLDS[k],
                    cfg.WSOVOD.SAMPLING.IOU_LABELS[k],
                    allow_low_quality_matches=False,
                )
        batch_size_per_images = cfg.WSOVOD.SAMPLING.BATCH_SIZE_PER_IMAGE
        positive_sample_fractions = cfg.WSOVOD.SAMPLING.POSITIVE_FRACTION

        cls_agnostic_bbox_known = cfg.WSOVOD.CLS_AGNOSTIC_BBOX_KNOWN
        output_dir = cfg.OUTPUT_DIR
        vis_test = cfg.VIS_TEST
        vis_period = cfg.VIS_PERIOD

        rpn_on = False if cfg.MODEL.PROPOSAL_GENERATOR.NAME == "PrecomputedProposals" else True

        metadatas = [MetadataCatalog.get(name) for name in cfg.DATASETS.MIXED_DATASETS.NAMES]

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "object_miners": object_miners,
            "num_classes_list": cfg.DATASETS.MIXED_DATASETS.NUM_CLASSES,
            "output_dir": output_dir,
            "vis_test": vis_test,
            "vis_period": vis_period,
            "mrrp_on": mrrp_on,
            "mrrp_num_branch": mrrp_num_branch,
            "mrrp_fast": mrrp_fast,
            "refine_K": refine_K,
            "refine_mist": refine_mist,
            "refine_reg": refine_reg,
            "box_refinery": box_refinery,
            "sampling_on": sampling_on,
            "proposal_matchers": proposal_matchers,
            "batch_size_per_images": batch_size_per_images,
            "positive_sample_fractions": positive_sample_fractions,
            "cls_agnostic_bbox_known": cls_agnostic_bbox_known,
            "pooler_type": pooler_type,
            "rpn_on": rpn_on,
            "metadatas": metadatas,
        }

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        data_aware_features = None,
        targets: Optional[List[Instances]] = None,
        classifier = None,
        source_id = 0,
        append_background = True,
        file_names = None,
        loaded_proposals = None,
    ):
        """
        See :class:`ROIHeads.forward`.
        """

        if self.training:
            self.num_classes = self.num_classes_list[source_id]
            self.metadata = self.metadatas[source_id]
            self.gt_classes_img, self.gt_classes_img_int, self.gt_classes_img_oh = get_image_level_gt(
                targets, self.num_classes
            )

        self.images = images
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
            self._vis_proposal(proposals, "train", "_proposals")
        else:
            self._save_proposal_test(proposals, "test", "_proposals")

        del targets

        if self.training:
            losses = self._forward_box(
                features, 
                proposals, 
                data_aware_features, 
                classifier,
                append_background, 
                source_id = source_id,
                file_names=file_names,
                loaded_proposals=loaded_proposals,
            )
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.

            self.iter = self.iter + 1
            if self.iter_test > 0:
                self.epoch_test = self.epoch_test + 1
            self.iter_test = 0

            return proposals, losses
        else:
            pred_instances, all_scores, all_boxes = self._forward_box(features, proposals, data_aware_features, classifier,append_background)
            
            self.iter_test = self.iter_test + 1

            return pred_instances, {}, all_scores, all_boxes

    def _forward_box(
        self, 
        features: Dict[str, torch.Tensor], 
        proposals: List[Instances], 
        data_aware_features = None, 
        classifier = None,
        append_background = True,
        source_id = 0,
        file_names = None,
        loaded_proposals = None,
    ):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        if self.mrrp_on:
            features = [torch.chunk(f, self.mrrp_num_branch) for f in features]
            features = [ff for f in features for ff in f]

        box_features = self.box_pooler(
            features,
            [x.proposal_boxes for x in proposals],
            level_ids=[torch.div(x.level_ids, 1000, rounding_mode='floor') for x in proposals] if self.mrrp_on else None,
        )

        objectness_logits = torch.cat([x.objectness_logits + 1 for x in proposals], dim=0)
        if self.pooler_type == "ROILoopPool":
            objectness_logits = torch.cat(
                [objectness_logits, objectness_logits, objectness_logits], dim=0
            )

        box_features = box_features * objectness_logits.view(-1, 1, 1, 1)
        if self.training and objectness_logits.numel() > 0:
            storage = get_event_storage()
            storage.put_scalar("proposals/objectness_logits+1 mean", objectness_logits.mean())
            storage.put_scalar("proposals/objectness_logits+1 max", objectness_logits.max())
            storage.put_scalar("proposals/objectness_logits+1 min", objectness_logits.min())

        box_features = self.box_head(box_features)

        if self.pooler_type == "ROILoopPool":
            box_features, box_features_frame, box_features_context = torch.chunk(
                box_features, 3, dim=0
            )
            if data_aware_features is not None:
                box_features = box_features + data_aware_features
                box_features_frame = box_features_frame + data_aware_features
                box_features_context = box_features_context + data_aware_features
            predictions = self.object_miners[source_id](
                [box_features, box_features_frame, box_features_context], proposals, context=True
            )
            del box_features_frame
            del box_features_context
        else:
            if data_aware_features is not None:
                box_features += data_aware_features
            predictions = self.object_miners[source_id](box_features, proposals)

        if self.training:
            losses = self.object_miners[source_id].losses(predictions, proposals, self.gt_classes_img_oh)

            self.pred_class_img_logits = (
                self.object_miners[source_id].predict_probs_img(predictions, proposals).clone().detach()
            )

            prev_pred_scores = self.object_miners[source_id].predict_probs(predictions, proposals)
            prev_pred_scores = [prev_pred_score.detach() for prev_pred_score in prev_pred_scores]
            prev_pred_boxes = self.object_miners[source_id].predict_boxes(predictions, proposals)
            self._vis_prediction(
                prev_pred_boxes,
                prev_pred_scores,
                proposals,
                top_k=100,
                prefix="train",
                suffix="_ObjectMining",
            )
            if self.sam:
                self.sam.reset_buffer()
            for k in range(self.refine_K):
                suffix = "_r" + str(k)
                term_weight = 1
                if self.refine_mist:
                    targets = self.get_pgt_mist(
                        prev_pred_boxes, 
                        prev_pred_scores, 
                        proposals, 
                        sam=self.sam if self.refine_reg[k] else None, 
                        file_names=file_names
                    )
                    self._vis_pgt(targets, "pgt_mist", suffix)
                    if k == 0:
                        term_weight = 1
                else:
                    targets = self.get_pgt_top_k(
                        prev_pred_boxes, 
                        prev_pred_scores, 
                        proposals, 
                        sam=self.sam if self.refine_reg[k] else None, 
                        file_names=file_names
                    )
                    self._vis_pgt(targets, "pgt_top_k", suffix)

                if self.sampling_on:
                    proposals_k = self.label_and_sample_proposals_wsl(
                        k, proposals, targets, suffix=suffix
                    )
                else:
                    proposals_k = self.label_and_sample_proposals(proposals, targets, suffix=suffix)

                predictions_k = self.box_refinery[k](box_features, classifier = classifier, append_background=append_background)

                losses_k = self.box_refinery[k].losses(predictions_k, proposals_k,self.num_classes)
                for loss_name in losses_k.keys():
                    losses_k[loss_name] = losses_k[loss_name] * term_weight

                prev_pred_scores = self.box_refinery[k].predict_probs(predictions_k, proposals_k)
                prev_pred_boxes = self.box_refinery[k].predict_boxes(predictions_k, proposals_k)
                prev_pred_scores = [
                    prev_pred_score.detach() for prev_pred_score in prev_pred_scores
                ]
                prev_pred_boxes = [prev_pred_box.detach() for prev_pred_box in prev_pred_boxes]

                self._vis_prediction(
                    prev_pred_boxes,
                    prev_pred_scores,
                    proposals,
                    top_k=100,
                    prefix="train",
                    suffix=suffix,
                )

                losses.update(losses_k)

            if self.rpn_on:
                if loaded_proposals and False:
                    keeps = [
                        batched_nms(
                            prop.proposal_boxes.tensor, 
                            prop.objectness_logits, 
                            torch.zeros_like(prop.objectness_logits), 
                            0.2
                        )
                        for prop in loaded_proposals
                    ]
                    targets = [
                        Instances(
                            prop.image_size,
                            gt_boxes=prop.proposal_boxes[keep],
                            gt_scores=prop.objectness_logits[keep],
                            gt_classes=torch.zeros(prop.objectness_logits[keep].shape,dtype=torch.int),
                        )
                        for prop,keep in zip(loaded_proposals,keeps)
                    ]
                    self._vis_pgt(targets, "loaded_proposals", "_rpn")
                else:
                    # targets = self.get_pgt_mist(
                    #     prev_pred_boxes, 
                    #     prev_pred_scores, 
                    #     proposals, 
                    #     top_pro=0.05, 
                    #     # sam=self.sam, 
                    #     # file_names=file_names
                    # )
                    # self._vis_pgt(targets, "get_pgt_mist", "_rpn")
                    targets = self.get_pgt_top_k(
                        prev_pred_boxes, 
                        prev_pred_scores, 
                        proposals, 
                        top_k=1, 
                        sam=self.sam, 
                        file_names=file_names
                    )
                    self._vis_pgt(targets, "pgt_top_k", "_rpn")
                self.proposal_targets = targets

            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.object_miners[0].predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        
        else:
            if self.refine_K > 0:
                predictions_K = []
                for k in range(self.refine_K):
                    predictions_k = self.box_refinery[k](box_features, classifier, append_background)
                    predictions_K.append(predictions_k)
                pred_instances, _, all_scores, all_boxes = self.box_refinery[-1].inference(
                    predictions_K, proposals
                )
            else:
                predictions = self.object_miners[0](box_features, proposals, context=True)
                pred_instances, _, all_scores, all_boxes = self.box_predictor.inference(
                    predictions, proposals
                )
            return pred_instances, all_scores, all_boxes

    @torch.no_grad()
    def get_pgt_mist(
        self, 
        prev_pred_boxes, 
        prev_pred_scores, 
        proposals, 
        top_pro=0.15,
        sam=None,
        file_names=None
    ):
        pgt_scores, pgt_boxes, pgt_classes, pgt_weights = self.get_pgt_top_k(
            prev_pred_boxes,
            prev_pred_scores,
            proposals,
            top_k=top_pro,
            thres=0.05,
            # thres=0.0,
            need_instance=False,
            need_weight=True,
        )

        # NMS
        pgt_idxs = [torch.zeros_like(pgt_class) for pgt_class in pgt_classes]
        keeps = [
            batched_nms(pgt_box, pgt_score, pgt_class, 0.2)
            for pgt_box, pgt_score, pgt_class in zip(pgt_boxes, pgt_scores, pgt_idxs)
        ]
        pgt_scores = [pgt_score[keep] for pgt_score, keep in zip(pgt_scores, keeps)]
        pgt_boxes = [pgt_box[keep] for pgt_box, keep in zip(pgt_boxes, keeps)]
        pgt_classes = [pgt_class[keep] for pgt_class, keep in zip(pgt_classes, keeps)]
        pgt_weights = [pgt_weight[keep] for pgt_weight, keep in zip(pgt_weights, keeps)]

        # sam refine
        pgt_boxes_old = [None for _ in range(len(self.images))]
        polygons_masks_per_image = [None for _ in range(len(self.images))]
        if sam:
            pgt_boxes_old = [Boxes(pgt_box.clone()) for pgt_box in pgt_boxes]
            bitmasks_per_image = []
            for i,pgt_box in enumerate(pgt_boxes):
                center_x = (pgt_box[:,0] + pgt_box[:,2]) / 2
                center_y = (pgt_box[:,1] + pgt_box[:,3]) / 2
                width = pgt_box[:,2] - pgt_box[:,0]
                height = pgt_box[:,3] - pgt_box[:,1]
                width *= 1.1
                height *= 1.1
                # width *= 1.0
                # height *= 1.0
                new_x1 = center_x - width / 2
                new_y1 = center_y - height / 2
                new_x2 = center_x + width / 2
                new_y2 = center_y + height / 2
                x1 = new_x1.clamp(min=0, max=self.images[i].shape[-1])
                y1 = new_y1.clamp(min=0, max=self.images[i].shape[-2])
                x2 = new_x2.clamp(min=0, max=self.images[i].shape[-1])
                y2 = new_y2.clamp(min=0, max=self.images[i].shape[-2])
                input_boxes = torch.stack((x1, y1, x2, y2), dim=-1)
                transformed_boxes = sam.transform.apply_boxes_torch(input_boxes, self.images[i].shape[-2:])
                sam.set_image(
                    self.images[i].cpu().clone().numpy().astype(np.uint8).transpose(1,2,0),
                    image_format="BGR",
                    file_name=file_names[i],
                )
                bitmasks, mask_scores, _ = sam.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                    file_name=file_names[i],
                )
                bitmasks = bitmasks.squeeze(1)
                mask_scores = mask_scores.squeeze(1)
                bitmasks_per_image.append(bitmasks)

            def mask_to_polygons(mask):
                # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
                # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
                # Internal contours (holes) are placed in hierarchy-2.
                # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
                mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
                res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                hierarchy = res[-1]
                if hierarchy is None:  # empty mask
                    return [], False
                has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
                res = res[-2]
                res = [x.flatten() for x in res]
                # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
                # We add 0.5 to turn them into real-value coordinate space. A better solution
                # would be to first +0.5 and then dilate the returned polygon by 0.5.
                res = [x + 0.5 for x in res if len(x) >= 6]
                return res, has_holes

            polygons_masks_per_image = [
                    [mask_to_polygons(bitmask)[0] for bitmask in bitmasks.cpu().numpy()]
                    for bitmasks in bitmasks_per_image
            ]
            polygons_masks_per_image = [PolygonMasks(polygons_masks) for polygons_masks in polygons_masks_per_image]
            pgt_boxes = [polygons_masks.get_bounding_boxes().tensor.to(sam.device) for polygons_masks in polygons_masks_per_image]


        pgt_boxes = [Boxes(pgt_box) for pgt_box in pgt_boxes]

        if sam:
            targets = [
                Instances(
                    proposals[i].image_size,
                    gt_boxes=pgt_box,
                    ori_pgt_boxes=ori_pgt_box,
                    gt_masks=pgt_masks,
                    gt_classes=pgt_class,
                    gt_scores=pgt_score,
                    gt_weights=pgt_weight,
                )
                for i, (pgt_box, ori_pgt_box, pgt_masks, pgt_class, pgt_score, pgt_weight) in enumerate(
                    zip(pgt_boxes, pgt_boxes_old, polygons_masks_per_image, pgt_classes, pgt_scores, pgt_weights)
                )
            ]
        else:
            targets = [
                Instances(
                    proposals[i].image_size,
                    gt_boxes=pgt_box,
                    gt_classes=pgt_class,
                    gt_scores=pgt_score,
                    gt_weights=pgt_weight,
                )
                for i, (pgt_box, pgt_class, pgt_score, pgt_weight) in enumerate(
                    zip(pgt_boxes, pgt_classes, pgt_scores, pgt_scores)
                )
            ]

        return targets

    @torch.no_grad()
    def get_pgt_top_k(
        self,
        prev_pred_boxes,
        prev_pred_scores,
        proposals,
        top_k=1,
        thres=0,
        need_instance=True,
        need_weight=True,
        sam=None,
        file_names=None,
    ):
        assert isinstance(prev_pred_boxes, tuple) or isinstance(prev_pred_boxes, list)

        assert isinstance(prev_pred_boxes[0], torch.Tensor)
        num_preds = [prev_pred_box.size(0) for prev_pred_box in prev_pred_boxes]
        if prev_pred_boxes[0].size(1) == 4:
            prev_pred_boxes = [
                prev_pred_box.unsqueeze(1).expand(num_pred, self.num_classes, 4)
                for num_pred, prev_pred_box in zip(num_preds, prev_pred_boxes)
            ]
        else:
            assert (prev_pred_boxes[0].size(1) == self.num_classes * 4) or (prev_pred_boxes[0].size(1) == self.num_classes and prev_pred_boxes[0].size(2) == 4)
            prev_pred_boxes = [
                prev_pred_box.view(-1, self.num_classes, 4) for prev_pred_box in prev_pred_boxes
            ]

        if isinstance(prev_pred_scores, torch.Tensor):
            num_preds_per_image = [len(p) for p in proposals]
            prev_pred_scores = prev_pred_scores.split(num_preds_per_image, dim=0)
        else:
            assert isinstance(prev_pred_scores, tuple) or isinstance(
                prev_pred_scores, list
            ), prev_pred_scores
            assert isinstance(prev_pred_scores[0], torch.Tensor), prev_pred_scores[0]

        prev_pred_scores = [
            torch.index_select(prev_pred_score, 1, gt_int)
            for prev_pred_score, gt_int in zip(prev_pred_scores, self.gt_classes_img_int)
        ]
        prev_pred_boxes = [
            torch.index_select(prev_pred_box, 1, gt_int)
            for prev_pred_box, gt_int in zip(prev_pred_boxes, self.gt_classes_img_int)
        ]


        # filter small pgt
        def get_area(box):
            return (box[:,:,2]-box[:,:,0])*(box[:,:,3]-box[:,:,1])
        
        prev_pred_boxes_keep = [
            get_area(box)>20
            for box in prev_pred_boxes
        ]

        prev_pred_boxes = [
            boxes.masked_select(
                torch.unsqueeze(mask, 2).expand(-1, gt_int.numel(), 4)
            ).view(-1,gt_int.numel(), 4)
            for boxes, mask, gt_int in zip(
                prev_pred_boxes, prev_pred_boxes_keep, self.gt_classes_img_int
            )
        ]
        prev_pred_scores = [
            scores.masked_select(mask).view(-1, gt_int.numel())
            for scores, mask, gt_int in zip(
                prev_pred_scores, prev_pred_boxes_keep, self.gt_classes_img_int
            )
        ]


        # get top k
        num_preds = [prev_pred_score.size(0) for prev_pred_score in prev_pred_scores]
        if top_k >= 1:
            top_ks = [min(num_pred, int(top_k)) for num_pred in num_preds]
        elif top_k < 1 and top_k > 0:
            top_ks = [max(int(num_pred * top_k), 1) for num_pred in num_preds]
        else:
            top_ks = [min(num_pred, 1) for num_pred in num_preds]
        pgt_scores_idxs = [
            torch.topk(prev_pred_score, top_k, dim=0)
            for prev_pred_score, top_k in zip(prev_pred_scores, top_ks)
        ]
        pgt_scores = [item[0] for item in pgt_scores_idxs]
        pgt_idxs = [item[1] for item in pgt_scores_idxs]
        pgt_idxs = [
            torch.unsqueeze(pgt_idx, 2).expand(top_k, gt_int.numel(), 4)
            for pgt_idx, top_k, gt_int in zip(pgt_idxs, top_ks, self.gt_classes_img_int)
        ]
        pgt_boxes = [
            torch.gather(prev_pred_box, 0, pgt_idx)
            for prev_pred_box, pgt_idx in zip(prev_pred_boxes, pgt_idxs)
        ]
        pgt_classes = [
            torch.unsqueeze(gt_int, 0).expand(top_k, gt_int.numel())
            for gt_int, top_k in zip(self.gt_classes_img_int, top_ks)
        ]
        if need_weight:
            pgt_weights = [
                torch.index_select(pred_logits, 1, gt_int).expand(top_k, gt_int.numel())
                for pred_logits, gt_int, top_k in zip(
                    self.pred_class_img_logits.split(1, dim=0), self.gt_classes_img_int, top_ks
                )
            ]

        if thres > 0:
            # get large scores
            masks = [pgt_score.ge(thres) for pgt_score in pgt_scores]
            masks = [
                torch.cat([torch.full_like(mask[0:1, :], True), mask[1:, :]], dim=0)
                for mask in masks
            ]
            pgt_scores = [
                torch.masked_select(pgt_score, mask) for pgt_score, mask in zip(pgt_scores, masks)
            ]
            pgt_boxes = [
                torch.masked_select(
                    pgt_box, torch.unsqueeze(mask, 2).expand(top_k, gt_int.numel(), 4)
                )
                for pgt_box, mask, top_k, gt_int in zip(
                    pgt_boxes, masks, top_ks, self.gt_classes_img_int
                )
            ]
            pgt_classes = [
                torch.masked_select(pgt_class, mask) for pgt_class, mask in zip(pgt_classes, masks)
            ]
            if need_weight:
                pgt_weights = [
                    torch.masked_select(pgt_weight, mask)
                    for pgt_weight, mask in zip(pgt_weights, masks)
                ]

        pgt_scores = [pgt_score.reshape(-1) for pgt_score in pgt_scores]
        pgt_boxes = [pgt_box.reshape(-1, 4) for pgt_box in pgt_boxes]
        pgt_classes = [pgt_class.reshape(-1) for pgt_class in pgt_classes]
        if need_weight:
            pgt_weights = [pgt_weight.reshape(-1) for pgt_weight in pgt_weights]

            pgt_weights = [
                pgt_weight
                if pgt_weight.numel() > 0
                else torch.tensor([1], dtype=pgt_weight.dtype, device=pgt_weight.device)
                for pgt_weight in pgt_weights
            ]

        pgt_scores = [
            pgt_score
            if pgt_score.numel() > 0
            else torch.tensor([1], dtype=pgt_score.dtype, device=pgt_score.device)
            for pgt_score in pgt_scores
        ]
        pgt_boxes = [
            pgt_box
            if pgt_box.numel() > 0
            else torch.tensor(
                [[-10000, -10000, 10000, 10000]], dtype=pgt_box.dtype, device=pgt_box.device
            )
            for pgt_box in pgt_boxes
        ]
        pgt_classes = [
            pgt_class
            if pgt_class.numel() > 0
            else torch.tensor([0], dtype=pgt_class.dtype, device=pgt_class.device)
            for pgt_class in pgt_classes
        ]

        if not need_instance:
            if need_weight:
                return pgt_scores, pgt_boxes, pgt_classes, pgt_weights
            else:
                return pgt_scores, pgt_boxes, pgt_classes

        # sam refine
        pgt_boxes_old = [None for _ in range(len(self.images))]
        polygons_masks_per_image = [None for _ in range(len(self.images))]
        if sam:
            pgt_boxes_old = [Boxes(pgt_box.clone()) for pgt_box in pgt_boxes]
            bitmasks_per_image = []
            for i,pgt_box in enumerate(pgt_boxes):
                center_x = (pgt_box[:,0] + pgt_box[:,2]) / 2
                center_y = (pgt_box[:,1] + pgt_box[:,3]) / 2
                width = pgt_box[:,2] - pgt_box[:,0]
                height = pgt_box[:,3] - pgt_box[:,1]
                width *= 1.1
                height *= 1.1
                # width *= 1.0
                # height *= 1.0
                new_x1 = center_x - width / 2
                new_y1 = center_y - height / 2
                new_x2 = center_x + width / 2
                new_y2 = center_y + height / 2
                x1 = new_x1.clamp(min=0, max=self.images[i].shape[-1])
                y1 = new_y1.clamp(min=0, max=self.images[i].shape[-2])
                x2 = new_x2.clamp(min=0, max=self.images[i].shape[-1])
                y2 = new_y2.clamp(min=0, max=self.images[i].shape[-2])
                input_boxes = torch.stack((x1, y1, x2, y2), dim=-1)
                transformed_boxes = sam.transform.apply_boxes_torch(input_boxes, self.images[i].shape[-2:])
                sam.set_image(
                    self.images[i].cpu().clone().numpy().astype(np.uint8).transpose(1,2,0),
                    image_format="BGR",
                    file_name=file_names[i],
                )
                bitmasks, mask_scores, _ = sam.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                    file_name=file_names[i],
                )
                bitmasks = bitmasks.squeeze(1)
                mask_scores = mask_scores.squeeze(1)
                bitmasks_per_image.append(bitmasks)

            def mask_to_polygons(mask):
                # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
                # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
                # Internal contours (holes) are placed in hierarchy-2.
                # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
                mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
                res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                hierarchy = res[-1]
                if hierarchy is None:  # empty mask
                    return [], False
                has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
                res = res[-2]
                res = [x.flatten() for x in res]
                # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
                # We add 0.5 to turn them into real-value coordinate space. A better solution
                # would be to first +0.5 and then dilate the returned polygon by 0.5.
                res = [x + 0.5 for x in res if len(x) >= 6]
                return res, has_holes

            polygons_masks_per_image = [
                [mask_to_polygons(bitmask)[0] for bitmask in bitmasks.cpu().numpy()]
                for bitmasks in bitmasks_per_image
            ]
            polygons_masks_per_image = [PolygonMasks(polygons_masks) for polygons_masks in polygons_masks_per_image]
            for i,polygons_masks in enumerate(polygons_masks_per_image):
                pgt_box = polygons_masks.get_bounding_boxes().tensor.to(sam.device)
                inf_indices = torch.any(pgt_box == float('inf'), dim=1).nonzero(as_tuple=True)[0]
                pgt_box[inf_indices] = pgt_boxes[i][inf_indices]
                pgt_boxes[i] = pgt_box

        pgt_boxes = [Boxes(pgt_box) for pgt_box in pgt_boxes]
        if sam:
            if need_weight:
                targets = [
                    Instances(
                        proposals[i].image_size,
                        gt_boxes=pgt_box,
                        ori_pgt_boxes=ori_pgt_box,
                        gt_masks=pgt_masks,
                        gt_classes=pgt_class,
                        gt_scores=pgt_score,
                        gt_weights=pgt_weight,
                    )
                    for i, (pgt_box, ori_pgt_box, pgt_masks, pgt_class, pgt_score, pgt_weight) in enumerate(
                        zip(pgt_boxes, pgt_boxes_old, polygons_masks_per_image, pgt_classes, pgt_scores, pgt_weights)
                    )
                ]
            else:
                targets = [
                    Instances(
                        proposals[i].image_size,
                        gt_boxes=pgt_box,
                        pgt_boxes=ori_pgt_box,
                        gt_classes=pgt_class,
                        gt_scores=pgt_score,
                    )
                    for i, (pgt_box, ori_pgt_box, pgt_class, pgt_score) in enumerate(
                        zip(pgt_boxes, pgt_boxes_old, pgt_classes, pgt_scores)
                    )
                ]
        else:
            if need_weight:
                targets = [
                    Instances(
                        proposals[i].image_size,
                        gt_boxes=pgt_box,
                        gt_classes=pgt_class,
                        gt_scores=pgt_score,
                        gt_weights=pgt_weight,
                    )
                    for i, (pgt_box, pgt_class, pgt_score, pgt_weight) in enumerate(
                        zip(pgt_boxes, pgt_classes, pgt_scores, pgt_weights)
                    )
                ]
            else:
                targets = [
                    Instances(
                        proposals[i].image_size,
                        gt_boxes=pgt_box,
                        gt_classes=pgt_class,
                        gt_scores=pgt_score,
                    )
                    for i, (pgt_box, pgt_class, pgt_score) in enumerate(
                        zip(pgt_boxes, pgt_classes, pgt_scores)
                    )
                ]            

        return targets

    @torch.no_grad()
    def _vis_pgt(self, targets, prefix, suffix, thickness=4):
        if self.vis_period <= 0 or self.iter % self.vis_period > 0:
            return
        storage = get_event_storage()

        targets = copy.deepcopy(targets)

        output_dir = os.path.join(self.output_dir, prefix)
        if self.iter == 0:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        for b, target in enumerate(targets):
            img = self.images.tensor[b, ...].clone().detach() * self.pixel_std + self.pixel_mean
            img = img.cpu().numpy().transpose((1, 2, 0))
            img = img.astype(np.uint8)
            h, w = img.shape[:2]
            img = img.copy()


            # if target.has("gt_scores"):
            #     target.scores = target.gt_scores.clone().detach().cpu()
            # if target.has("gt_boxes"):
            #     target.pred_boxes = target.gt_boxes.tensor.clone().detach().cpu()
            # if target.has("gt_classes"):
            #     target.pred_classes = target.gt_classes.clone().detach().cpu()

            vis = Visualizer(img, self.metadata)
            # vis._default_font_size = max(
            #     np.sqrt(h * w) // 40, 20
            # )
            # vis_pred = vis.draw_instance_predictions(target).get_image()

            def _create_text_labels(classes, scores, class_names, is_crowd=None):
                """
                Args:
                    classes (list[int] or None):
                    scores (list[float] or None):
                    class_names (list[str] or None):
                    is_crowd (list[bool] or None):
                Returns:
                    list[str] or None
                """
                labels = None
                if classes is not None:
                    if class_names is not None and len(class_names) > 0:
                        labels = [class_names[i] for i in classes]
                    else:
                        labels = [str(i) for i in classes]
                if scores is not None:
                    if labels is None:
                        labels = ["{:.0f}%".format(s * 100) for s in scores]
                    else:
                        labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
                if labels is not None and is_crowd is not None:
                    labels = [l + ("|crowd" if crowd else "") for l, crowd in zip(labels, is_crowd)]
                return labels
            
            vis_pred = vis.overlay_instances(
                boxes=target.gt_boxes.tensor.clone().detach().cpu(),
                labels=_create_text_labels(target.gt_classes,target.gt_scores,self.metadata.thing_classes) if target.has('gt_classes') else None,
                masks=target.gt_masks if target.has('gt_masks') else None,
            ).get_image()
            if target.has('ori_pgt_boxes'):
                v_gt_ori = Visualizer(img, self.metadata)
                v_gt_ori = v_gt_ori.overlay_instances(
                    boxes=target.ori_pgt_boxes.tensor.clone().detach().cpu(),
                    labels=_create_text_labels(target.gt_classes,target.gt_scores,self.metadata.thing_classes)
                )
                vis_pred = np.concatenate((v_gt_ori.get_image(), vis_pred), axis=1)

            device_index = comm.get_rank()
            save_name = (
                "i" + str(self.iter) + "_g" + str(device_index) + "_b" + str(b) + suffix + ".png"
            )

            save_path = os.path.join(output_dir, save_name)
            cv2.imwrite(save_path, vis_pred)

            vis_pred = vis_pred.transpose(2, 0, 1)
            vis_name = prefix + "_g" + str(device_index) + "_b" + str(b) + suffix
            storage.put_image(vis_name, vis_pred)

    @torch.no_grad()
    def _vis_prediction(
        self,
        prev_pred_boxes,
        prev_pred_scores,
        proposals,
        top_k=1,
        thres=0.01,
        thickness=4,
        prefix="",
        suffix="",
    ):
        if self.vis_period <= 0 or self.iter % self.vis_period > 0:
            return

        targets = self.get_pgt_top_k(
            prev_pred_boxes,
            prev_pred_scores,
            proposals,
            top_k=top_k,
            thres=thres,
            need_weight=False,
        )

        self._vis_pgt(targets, prefix, suffix, thickness)

    @torch.no_grad()
    def _vis_proposal(self, proposals, prefix, suffix):
        if self.vis_period <= 0 or self.iter % self.vis_period > 0:
            return

        prev_pred_boxes = [p.proposal_boxes for p in proposals]
        num_preds = [len(prev_pred_box) for prev_pred_box in proposals]
        prev_pred_boxes = [
            prev_pred_box.tensor.unsqueeze(1).expand(num_pred, self.num_classes, 4)
            for num_pred, prev_pred_box in zip(num_preds, prev_pred_boxes)
        ]

        prev_pred_scores = [p.objectness_logits for p in proposals]
        prev_pred_scores = [
            prev_pred_score.unsqueeze(1).expand(num_pred, self.num_classes + 1)
            for num_pred, prev_pred_score in zip(num_preds, prev_pred_scores)
        ]

        self._vis_prediction(
            prev_pred_boxes,
            prev_pred_scores,
            proposals,
            top_k=2048,
            thres=-9999,
            thickness=1,
            prefix=prefix,
            suffix=suffix,
        )

    @torch.no_grad()
    def _save_proposal_test(self, proposals, prefix, suffix):
        if self.training or not self.vis_test:
            return

        output_dir = os.path.join(self.output_dir, prefix)
        if self.iter_test == 0:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        for b, p in enumerate(proposals):
            box = p.proposal_boxes.tensor.clone().detach().cpu().numpy()
            logit = p.objectness_logits.clone().detach().cpu().numpy()
            level_ids = p.level_ids.clone().detach().cpu().numpy()

            gpu_id = p.objectness_logits.device.index
            id_str = "i" + str(self.iter_test) + "_g" + str(gpu_id) + "_b" + str(b)

            save_path = os.path.join(output_dir, id_str + "_box" + suffix + ".npy")
            np.save(save_path, box)

            save_path = os.path.join(output_dir, id_str + "_logit" + suffix + ".npy")
            np.save(save_path, logit)

            save_path = os.path.join(output_dir, id_str + "_level" + suffix + ".npy")
            np.save(save_path, level_ids)

    @torch.no_grad()
    def _vis_box(
        self,
        prev_pred_boxes,
        prev_pred_scores,
        proposals,
        top_k=1,
        thres=0.01,
        thickness=4,
        prefix="",
        suffix="",
    ):
        if self.vis_period <= 0 or self.iter % self.vis_period > 0:
            return
        storage = get_event_storage()

        pgt_scores, pgt_boxes, pgt_classes = self.get_pgt_top_k(
            prev_pred_boxes,
            prev_pred_scores,
            proposals,
            top_k=top_k,
            thres=thres,
            need_instance=False,
            need_weight=False,
            suffix="",
        )

        output_dir = os.path.join(self.output_dir, prefix)
        if self.iter == 0:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        for b, pgt_box in enumerate(pgt_boxes):
            img = self.images.tensor[b, ...].clone().detach() * self.pixel_std + self.pixel_mean
            img = img.cpu().numpy().transpose((1, 2, 0))
            img = img.astype(np.uint8)
            h, w = img.shape[:2]
            img_pgt = img.copy()

            device_index = pgt_box.device.index
            save_name = (
                "i" + str(self.iter) + "_g" + str(device_index) + "_b" + str(b) + suffix + ".png"
            )
            pgt_box = pgt_box.cpu().numpy()
            for i in range(pgt_box.shape[0]):
                x0, y0, x1, y1 = pgt_box[i, :]
                x0 = int(max(x0, 0))
                y0 = int(max(y0, 0))
                x1 = int(min(x1, w))
                y1 = int(min(y1, h))
                cv2.rectangle(img_pgt, (x0, y0), (x1, y1), (0, 0, 255), thickness)

            save_path = os.path.join(output_dir, save_name)
            cv2.imwrite(save_path, img_pgt)

            img_pgt = img_pgt.transpose(2, 0, 1)
            vis_name = prefix + "_g" + str(device_index) + "_b" + str(b) + suffix
            storage.put_image(vis_name, img_pgt)

    def _sample_proposals_wsl(
        self, k, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
    ):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes,
            self.batch_size_per_images[k],
            self.positive_sample_fractions[k],
            self.num_classes,
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)

        gt_classes_sp = torch.full_like(gt_classes, -1)
        gt_classes_sp[sampled_idxs] = gt_classes[sampled_idxs]

        sampled_idxs = torch.arange(gt_classes.shape[0])
        return sampled_idxs, gt_classes_sp[sampled_idxs]

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_fraction, self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances], suffix=""
    ):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        num_ig_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt and not self.cls_agnostic_bbox_known:
                proposals_per_image.gt_boxes = targets_per_image.gt_boxes[
                    matched_idxs[sampled_idxs]
                ]
            if has_gt and targets_per_image.has("gt_scores"):
                proposals_per_image.gt_scores = targets_per_image.gt_scores[
                    matched_idxs[sampled_idxs]
                ]
            if has_gt and targets_per_image.has("gt_weights"):
                proposals_per_image.gt_weights = targets_per_image.gt_weights[
                    matched_idxs[sampled_idxs]
                ]

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            num_ig_samples.append((gt_classes == -1).sum().item())
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1] - num_ig_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples" + suffix, np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples" + suffix, np.mean(num_bg_samples))
        storage.put_scalar("roi_head/num_ig_samples" + suffix, np.mean(num_ig_samples))

        return proposals_with_gt

    @torch.no_grad()
    def label_and_sample_proposals_wsl(
        self, k: int, proposals: List[Instances], targets: List[Instances], suffix=""
    ):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        num_ig_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matchers[k](match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals_wsl(
                k, matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt and not self.cls_agnostic_bbox_known:
                proposals_per_image.gt_boxes = targets_per_image.gt_boxes[
                    matched_idxs[sampled_idxs]
                ]
            if has_gt and targets_per_image.has("gt_scores"):
                proposals_per_image.gt_scores = targets_per_image.gt_scores[
                    matched_idxs[sampled_idxs]
                ]
            if has_gt and targets_per_image.has("gt_weights"):
                proposals_per_image.gt_weights = targets_per_image.gt_weights[
                    matched_idxs[sampled_idxs]
                ]
            if has_gt and targets_per_image.has("gt_masks"):
                proposals_per_image.gt_masks = targets_per_image.gt_masks[
                    matched_idxs[sampled_idxs]
                ]

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            num_ig_samples.append((gt_classes == -1).sum().item())
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1] - num_ig_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples" + suffix, np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples" + suffix, np.mean(num_bg_samples))
        storage.put_scalar("roi_head/num_ig_samples" + suffix, np.mean(num_ig_samples))

        return proposals_with_gt

    def get_features(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        data_aware_features = None,
    ):
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])

        objectness_logits = torch.cat([x.objectness_logits + 1 for x in proposals], dim=0)
        if self.pooler_type == "ROILoopPool":
            objectness_logits = torch.cat(
                [objectness_logits, objectness_logits, objectness_logits], dim=0
            )
        box_features = box_features * objectness_logits.view(-1, 1, 1, 1)
        box_features = self.box_head(box_features)
        
        if self.pooler_type == "ROILoopPool":
            box_features, box_features_frame, box_features_context = torch.chunk(
                box_features, 3, dim=0
            )
            if data_aware_features is not None:
                box_features = box_features + data_aware_features
                box_features_frame = box_features_frame + data_aware_features
                box_features_context = box_features_context + data_aware_features
            del box_features_frame
            del box_features_context
        else:
            if data_aware_features is not None:
                box_features += data_aware_features
        return box_features