# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
from detectron2.config import configurable
from detectron2.layers import Conv2d, Linear, ShapeSpec, get_norm
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.events import get_event_storage
from torch import nn
from torch.nn import functional as F
import logging


class DataAwareFeaturesHead(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        datasets_prototype_num: int = 5,
        features_dim: int = 512,
        cls_in_features: List[str],
        mrrp_on: bool = False,
        mrrp_num_branch: int = 3,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature.
            conv_dims (list[int]): the output dimensions of the conv layers
            fc_dims (list[int]): the output dimensions of the fc layers
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__()

        self.in_features = self.cls_in_features = cls_in_features
        self.features_dim = features_dim
        self.mrrp_on = mrrp_on
        self.mrrp_num_branch = mrrp_num_branch

        self.in_channels = [input_shape[f].channels for f in self.in_features]
        in_channels = [input_shape[f].channels for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.strides = [input_shape[f].stride for f in self.in_features]

        self._output_size = (in_channels,)

        self.datasets_prototype_num = datasets_prototype_num
        self.datasets_feat = nn.Embedding(self.datasets_prototype_num, self.features_dim)
        
        self.GAP = nn.AdaptiveAvgPool2d(1)

        fc_dims = [in_channels//16,self.datasets_prototype_num]

        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):
            fc = Linear(np.prod(self._output_size), fc_dim)
            self.fcs.append(fc)
            self.add_module("linear{}".format(k + 1), fc)
            if k < len(fc_dims) - 1:
                relu = nn.ReLU(inplace=True)
                self.fcs.append(relu)
                self.add_module("linear_relu{}".format(k + 1), relu)
            else:
                tanh = nn.Tanh()
                self.fcs.append(tanh)
                self.add_module("linear_tanh{}".format(k + 1), tanh)
            self._output_size = fc_dim

        for layer in self.fcs:
            if not isinstance(layer, Linear):
                continue
            nn.init.uniform_(layer.weight, -0.01, 0.01)
            torch.nn.init.constant_(layer.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        # fmt: off
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        mrrp_on = cfg.MODEL.MRRP.MRRP_ON
        mrrp_num_branch = cfg.MODEL.MRRP.NUM_BRANCH
        datasets_prototype_num = cfg.MODEL.ROI_BOX_HEAD.OPEN_VOCABULARY.PROTOTYPE_NUM
        # fmt: on
        return {
            "cls_in_features": in_features,
            "input_shape": input_shape,
            "datasets_prototype_num": datasets_prototype_num,
            "features_dim": cfg.MODEL.ROI_BOX_HEAD.DAN_DIM[-1],
            "mrrp_on": mrrp_on,
            "mrrp_num_branch": mrrp_num_branch,
        }

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        proposals,
    ):

        features = [features[f] for f in self.cls_in_features]
        if self.mrrp_on:
            features = [torch.stack(torch.chunk(f, self.mrrp_num_branch)).mean(0) for f in features]
        data_features = [self._forward(f) for f in features]
        if len(data_features)>1:
            data_features = torch.stack(data_features).mean(0)
        else:
            data_features = data_features[0]
        results = []
        for i in range(len(proposals)):
            results.append(data_features[i].repeat(len(proposals[i]),1))
        results = torch.cat(results)
        return results

    def _forward(self, x):
        x = self.GAP(x)
        x = x.flatten(start_dim=1)
        for k, layer in enumerate(self.fcs):
            x = layer(x)
        combined = torch.matmul(x, self.datasets_feat.weight)
        return combined


