# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn.functional as F
from torch import nn

import fvcore.nn.weight_init as weight_init
from detectron2.layers import Conv2d, FrozenBatchNorm2d, ShapeSpec
from detectron2.modeling.backbone.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY

from .mrrp_conv import MRRPConv

__all__ = ["build_mrrp_vgg_backbone"]


class PlainBlockBase(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            in_channels (int):
            out_channels (int):
            stride (int):
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self


class PlainBlock(PlainBlockBase):
    def __init__(self, in_channels, out_channels, num_conv=3, dilation=1, stride=1, has_pool=False):
        super().__init__(in_channels, out_channels, stride)

        self.num_conv = num_conv
        self.dilation = dilation

        self.has_pool = has_pool
        self.pool_stride = stride

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1 * dilation,
            bias=True,
            groups=1,
            dilation=dilation,
            norm=None,
        )
        weight_init.c2_msra_fill(self.conv1)

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1 * dilation,
            bias=True,
            groups=1,
            dilation=dilation,
            norm=None,
        )
        weight_init.c2_msra_fill(self.conv2)

        if self.num_conv > 2:
            self.conv3 = Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1 * dilation,
                bias=True,
                groups=1,
                dilation=dilation,
                norm=None,
            )
            weight_init.c2_msra_fill(self.conv3)

        if self.num_conv > 3:
            self.conv4 = Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1 * dilation,
                bias=True,
                groups=1,
                dilation=dilation,
                norm=None,
            )
            weight_init.c2_msra_fill(self.conv4)

        if self.has_pool:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=self.pool_stride, padding=0)

        assert num_conv < 5

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu_(x)

        x = self.conv2(x)
        x = F.relu_(x)

        if self.num_conv > 2:
            x = self.conv3(x)
            x = F.relu_(x)

        if self.num_conv > 3:
            x = self.conv4(x)
            x = F.relu_(x)

        if self.has_pool:
            x = self.pool(x)

        return x


class MRRPPlainBlock(PlainBlockBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_conv=3,
        stride=1,
        num_branch=3,
        dilations=(1, 2, 3),
        concat_output=False,
        test_branch_idx=-1,
        has_pool=False,
    ):
        super().__init__(in_channels, out_channels, stride)

        assert num_branch == len(dilations)

        self.num_branch = num_branch
        self.concat_output = concat_output
        self.test_branch_idx = test_branch_idx

        self.num_conv = num_conv

        self.has_pool = has_pool
        self.pool_stride = stride

        self.conv1 = MRRPConv(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            paddings=dilations,
            bias=True,
            groups=1,
            dilations=dilations,
            num_branch=num_branch,
            test_branch_idx=test_branch_idx,
            norm=None,
        )
        weight_init.c2_msra_fill(self.conv1)

        self.conv2 = MRRPConv(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            paddings=dilations,
            bias=True,
            groups=1,
            dilations=dilations,
            num_branch=num_branch,
            test_branch_idx=test_branch_idx,
            norm=None,
        )
        weight_init.c2_msra_fill(self.conv2)

        if self.num_conv > 2:
            self.conv3 = MRRPConv(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                paddings=1 * dilations,
                bias=True,
                groups=1,
                dilations=dilations,
                num_branch=num_branch,
                test_branch_idx=test_branch_idx,
                norm=None,
            )
            weight_init.c2_msra_fill(self.conv3)

        if self.num_conv > 3:
            self.conv4 = MRRPConv(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                paddings=1 * dilations,
                bias=True,
                groups=1,
                dilations=dilations,
                num_branch=num_branch,
                test_branch_idx=test_branch_idx,
                norm=None,
            )
            weight_init.c2_msra_fill(self.conv4)

        if self.has_pool:
            self.list_pool = []
            for _ in range(num_branch):
                self.list_pool.append(
                    nn.MaxPool2d(kernel_size=2, stride=self.pool_stride, padding=0)
                )
            self.list_pool = nn.ModuleList(self.list_pool)

        assert num_conv < 5

    def forward(self, x):
        num_branch = self.num_branch if self.training or self.test_branch_idx == -1 else 1
        if not isinstance(x, list):
            x = [x] * num_branch

        x = self.conv1(x)
        x = [F.relu_(b) for b in x]

        x = self.conv2(x)
        x = [F.relu_(b) for b in x]

        if self.num_conv > 2:
            x = self.conv3(x)
            x = [F.relu_(b) for b in x]

        if self.num_conv > 3:
            x = self.conv4(x)
            x = [F.relu_(b) for b in x]

        if self.has_pool:
            x = [p(b) for p, b in zip(self.list_pool, x)]

        if self.concat_output:
            x = torch.cat(x)

        return x


class VGG16(Backbone):
    def __init__(
        self,
        conv5_dilation,
        freeze_at,
        num_branch,
        branch_dilations,
        mrrp_stage,
        test_branch_idx,
        num_classes=None,
        out_features=None,
    ):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[ResNetBlock]]): several (typically 4) stages,
                each contains multiple :class:`ResNetBlockBase`.
            num_classes (None or int): if None, will not perform classification.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "linear", or "res2" ...
                If None, will return the output of the last layer.
        """
        super(VGG16, self).__init__()

        self.num_branch = num_branch
        self.branch_dilations = branch_dilations
        self.mrrp_stage = mrrp_stage
        self.test_branch_idx = test_branch_idx

        self.num_classes = num_classes

        self._out_feature_strides = {}
        self._out_feature_channels = {}

        self.stages_and_names = []

        name = "plain1"
        block = PlainBlock(3, 64, num_conv=2, stride=2, has_pool=True)
        blocks = [block]
        stage = nn.Sequential(*blocks)
        self.add_module(name, stage)
        self.stages_and_names.append((stage, name))
        self._out_feature_strides[name] = 2
        self._out_feature_channels[name] = blocks[-1].out_channels
        if freeze_at >= 1:
            for block in blocks:
                block.freeze()

        name = "plain2"
        block = PlainBlock(64, 128, num_conv=2, stride=2, has_pool=True)
        blocks = [block]
        stage = nn.Sequential(*blocks)
        self.add_module(name, stage)
        self.stages_and_names.append((stage, name))
        self._out_feature_strides[name] = 4
        self._out_feature_channels[name] = blocks[-1].out_channels
        if freeze_at >= 2:
            for block in blocks:
                block.freeze()

        name = "plain3"
        block = PlainBlock(128, 256, num_conv=3, stride=2, has_pool=True)
        blocks = [block]
        stage = nn.Sequential(*blocks)
        self.add_module(name, stage)
        self.stages_and_names.append((stage, name))
        self._out_feature_strides[name] = 8
        self._out_feature_channels[name] = blocks[-1].out_channels
        if freeze_at >= 3:
            for block in blocks:
                block.freeze()

        name = "plain4"
        block = PlainBlock(
            256, 512, num_conv=3, stride=1 if conv5_dilation == 2 else 2, has_pool=True
        )
        blocks = [block]
        stage = nn.Sequential(*blocks)
        self.add_module(name, stage)
        self.stages_and_names.append((stage, name))
        self._out_feature_strides[name] = 8 if conv5_dilation == 2 else 16
        self._out_feature_channels[name] = blocks[-1].out_channels
        if freeze_at >= 4:
            for block in blocks:
                block.freeze()

        name = "plain5"
        if name in self.mrrp_stage:
            block = MRRPPlainBlock(
                512,
                512,
                num_conv=3,
                stride=1,
                num_branch=self.num_branch,
                dilations=self.branch_dilations,
                concat_output=True,
                test_branch_idx=self.test_branch_idx,
                has_pool=False,
            )
        else:
            block = PlainBlock(
                512, 512, num_conv=3, stride=1, dilation=conv5_dilation, has_pool=False
            )
        blocks = [block]
        stage = nn.Sequential(*blocks)
        self.add_module(name, stage)
        self.stages_and_names.append((stage, name))
        self._out_feature_strides[name] = 8 if conv5_dilation == 2 else 16
        self._out_feature_channels[name] = blocks[-1].out_channels
        if freeze_at >= 5:
            for block in blocks:
                block.freeze()

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))

    def forward(self, x):
        outputs = {}
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x

        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


@BACKBONE_REGISTRY.register()
def build_mrrp_vgg_backbone(cfg, input_shape):

    # fmt: off
    depth                = cfg.MODEL.VGG.DEPTH
    conv5_dilation       = cfg.MODEL.VGG.CONV5_DILATION
    freeze_at            = cfg.MODEL.BACKBONE.FREEZE_AT
    num_branch           = cfg.MODEL.MRRP.NUM_BRANCH
    branch_dilations     = cfg.MODEL.MRRP.BRANCH_DILATIONS
    mrrp_stage        = cfg.MODEL.MRRP.MRRP_STAGE
    test_branch_idx      = cfg.MODEL.MRRP.TEST_BRANCH_IDX
    # fmt: on

    if depth == 16:
        return VGG16(
            conv5_dilation, freeze_at, num_branch, branch_dilations, mrrp_stage, test_branch_idx
        )