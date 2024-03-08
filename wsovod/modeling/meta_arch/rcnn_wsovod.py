# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.data import DatasetCatalog, MetadataCatalog
from torch import nn

from wsovod.modeling.roi_heads.roi_heads import build_roi_heads
from wsovod.modeling.class_heads.data_aware_features_head import DataAwareFeaturesHead
from wsovod.modeling.proposal_generator import WSOVODRPN_V2

from ..postprocessing import detector_postprocess

__all__ = ["GeneralizedRCNN_WSOVOD", ]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN_WSOVOD(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        cfg = None,
        backbone: Backbone,
        data_aware_head: nn.Module = None,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.data_aware_head = data_aware_head
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        
        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        self.logger = logging.getLogger(__name__)
        self.classifier = None

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        
        return {
            "cfg": cfg,
            "backbone": backbone,
            "data_aware_head": DataAwareFeaturesHead(cfg, backbone.output_shape()) if cfg.MODEL.ROI_BOX_HEAD.OPEN_VOCABULARY.DATA_AWARE else None,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.detach().cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]], classifier=None):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs, classifier = classifier)

        images = self.preprocess_image(batched_inputs)
        if "file_name" in batched_inputs[0]:
            file_names = [i['file_name'] for i in batched_inputs]
        else:
            file_names = None
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        loaded_proposals = None
        rpn_proposals = None
        if self.proposal_generator is not None:
            proposals, _ = self.proposal_generator(images, features, gt_instances)
            rpn_proposals = proposals
            if isinstance(self.proposal_generator, WSOVODRPN_V2):
                for p in proposals:
                    storage = get_event_storage()
                    cur_iter = storage.iter
                    p.objectness_logits = torch.sigmoid(p.objectness_logits)*(cur_iter/self.cfg.SOLVER.MAX_ITER)
            if "proposals" in batched_inputs[0]:
                proposals_ = [x["proposals"].to(self.device) for x in batched_inputs]
                loaded_proposals = proposals_
                for p1, p2 in zip(proposals, proposals_):
                    if not p1.has("level_ids"):
                        continue
                    low_id = torch.min(p1.level_ids)
                    high_id = torch.max(p1.level_ids) + 1
                    p2.level_ids = torch.randint(
                        low_id, high_id, (len(p2),), dtype=torch.int64, device=self.device
                    )

                proposals = [Instances.cat([p1, p2]) for p1, p2 in zip(proposals, proposals_)]
        else:
            assert "proposals" in batched_inputs[0]
            proposals_ = [x["proposals"].to(self.device) for x in batched_inputs]
            for p in proposals_:
                p.level_ids = torch.zeros((len(p),), dtype=torch.int64, device=self.device)
            proposals = proposals_
            loaded_proposals = proposals_

        if self.data_aware_head is not None:
            data_aware_features = self.data_aware_head(features,proposals)
        else:
            data_aware_features = None

        _, detector_losses = self.roi_heads(
            images, 
            features, 
            proposals, 
            data_aware_features, 
            gt_instances, 
            append_background = True, 
            file_names=file_names,
            loaded_proposals = loaded_proposals
        )


        if self.proposal_generator is not None:
            proposal_losses = self.proposal_generator.get_losses(self.roi_heads.proposal_targets)
        else:
            proposal_losses = {}
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, rpn_proposals if rpn_proposals else proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
        classifier = None,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:

            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
                if isinstance(self.proposal_generator, WSOVODRPN_V2):
                    for p in proposals:
                        p.objectness_logits = torch.sigmoid(p.objectness_logits)
                if "proposals" in batched_inputs[0]:
                    proposals_ = [x["proposals"].to(self.device) for x in batched_inputs]
                    for p1, p2 in zip(proposals, proposals_):
                        if not p1.has("level_ids"):
                            continue
                        low_id = torch.min(p1.level_ids)
                        high_id = torch.max(p1.level_ids) + 1
                        p2.level_ids = torch.randint(
                            low_id, high_id, (len(p2),), dtype=torch.int64, device=self.device
                        )

                    proposals = [Instances.cat([p1, p2]) for p1, p2 in zip(proposals, proposals_)]
            else:
                assert "proposals" in batched_inputs[0]
                proposals_ = [x["proposals"].to(self.device) for x in batched_inputs]
                for p in proposals_:
                    p.level_ids = torch.zeros((len(p),), dtype=torch.int64, device=self.device)
                proposals = proposals_

            if self.data_aware_head is not None:
                data_aware_features = self.data_aware_head(features,proposals)
            else:
                data_aware_features = None

            if classifier is not None:
                self.classifier = classifier
            elif self.classifier == None:
                weight_path = self.cfg.MODEL.ROI_BOX_HEAD.OPEN_VOCABULARY.WEIGHT_PATH_TEST
                self.logger.info("Loading " + weight_path)
                weight = (
                    torch.tensor(np.load(weight_path,encoding='bytes', allow_pickle=True), dtype=torch.float32).contiguous()
                )  # C x D
                self.logger.info(f"Loaded class weight {weight.size()}")
                self.classifier = weight
                self.classifier = self.classifier.to(self.cfg.MODEL.DEVICE)

            results, _, all_scores, all_boxes = self.roi_heads(images, features, proposals, data_aware_features, None, self.classifier, append_background = True)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results, all_scores, all_boxes = self.roi_heads.forward_with_given_boxes(
                features, detected_instances, data_aware_features
            )

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN_WSOVOD._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results, all_scores, all_boxes

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: Tuple[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results
