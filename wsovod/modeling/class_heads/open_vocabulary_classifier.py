# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from math import fabs
from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec

logger = logging.getLogger(__name__)


class OpenVocabularyClassifier(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        num_classes: int,
        weight_path: str,
        weight_dim: int = 512,
        use_bias: float = 0.0,
        norm_weight: bool = True,
        norm_temperature: float = 50.0,
    ):
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.norm_weight = norm_weight
        self.weight_dim = weight_dim
        self.norm_temperature = norm_temperature

        self.use_bias = fabs(use_bias)>1e-9
        if self.use_bias:
            self.cls_bias = nn.Parameter(torch.ones(1) * use_bias)

        self.projection = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, weight_dim),
            nn.ReLU(),
        )
        

        if weight_path == "rand":
            class_weight = torch.randn((weight_dim, num_classes))
            nn.init.normal_(class_weight, std=0.01)
        else:
            logger.info("Loading " + weight_path)
            class_weight = (
                torch.tensor(np.load(weight_path,encoding='bytes', allow_pickle=True), dtype=torch.float32)
                .permute(1, 0)
                .contiguous()
            )  # D x C
            logger.info(f"Loaded class weight {class_weight.size()}")

        if self.norm_weight:
            class_weight = F.normalize(class_weight, p=2, dim=0)

        if weight_path == "rand":
            self.class_weight = nn.Parameter(class_weight)
        else:
            self.register_buffer("class_weight", class_weight)

    @classmethod
    def from_config(cls, cfg, input_shape, weight_path = None, use_bias = None, norm_weight = None, norm_temperature = None):
        return {
            "input_shape": input_shape,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "weight_path": weight_path if weight_path is not None else cfg.MODEL.ROI_BOX_HEAD.OPEN_VOCABULARY.WEIGHT_PATH_TRAIN,
            "weight_dim": cfg.MODEL.ROI_BOX_HEAD.OPEN_VOCABULARY.WEIGHT_DIM,
            "use_bias": use_bias if use_bias is not None else cfg.MODEL.ROI_BOX_HEAD.OPEN_VOCABULARY.USE_BIAS,
            "norm_weight": norm_weight if norm_weight is not None else cfg.MODEL.ROI_BOX_HEAD.OPEN_VOCABULARY.NORM_WEIGHT,
            "norm_temperature": norm_temperature if norm_temperature is not None else cfg.MODEL.ROI_BOX_HEAD.OPEN_VOCABULARY.NORM_TEMP,
        }

    def forward(self, x, classifier=None, append_background = False):
        """
        Inputs:
            x: B x D
            classifier: (C', C' x D)
        """
        x = self.projection(x)

        if classifier is not None:
            class_weight = classifier.permute(1, 0).contiguous()  # D x C'
            class_weight = F.normalize(class_weight, p=2, dim=0) if self.norm_weight else class_weight
        else:
            class_weight = self.class_weight

        if self.norm_weight:
            x = self.norm_temperature * F.normalize(x, p=2, dim=1)
    
        if append_background:
            class_weight = torch.cat(
                [class_weight, class_weight.new_zeros((self.weight_dim, 1))], dim=1
            )  # D x (C + 1)
            # logger.info(f"Cated class weight {class_weight.size()}")

        x = torch.mm(x, class_weight)
        if self.use_bias:
            x = x + self.cls_bias
        return x