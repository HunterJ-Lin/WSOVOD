# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import datetime
import itertools
import logging
import math
import operator
import os
import tempfile
import time
import warnings
from collections import Counter

import detectron2.utils.comm as comm
import torch
from detectron2.engine.train_loop import HookBase
from detectron2.evaluation.testing import flatten_results_dict
from detectron2.solver import LRMultiplier
from detectron2.utils.events import EventStorage, EventWriter
from detectron2.utils.file_io import PathManager
from fvcore.common.checkpoint import Checkpointer
from fvcore.common.checkpoint import \
    PeriodicCheckpointer as _PeriodicCheckpointer
from fvcore.common.param_scheduler import ParamScheduler
from fvcore.common.timer import Timer
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

# __all__ = [
#     "CallbackHook",
#     "IterationTimer",
#     "PeriodicWriter",
#     "PeriodicCheckpointer",
#     "BestCheckpointer",
#     "LRScheduler",
#     "AutogradProfiler",
#     "EvalHook",
#     "PreciseBN",
#     "TorchProfiler",
#     "TorchMemoryStats",
# ]


"""
Implement some common hooks.
"""

class ParametersNormInspectHook(HookBase):

    def __init__(self, period, model, p):
        self._period = period
        self._model = model
        self._p = p
        logger = logging.getLogger(__name__)
        logger.info('period, norm '+str((period,p)))

    @torch.no_grad()
    def _do_inspect(self):
        results = {}
        # logger = logging.getLogger(__name__)
        for key,val in self._model.named_parameters(recurse=True):
            results[key] = torch.norm(val,p=self._p)
            self.trainer.storage.put_scalar('parameters norm {}/{}'.format(self._p,key),torch.norm(val,p=self._p))
        # logger.info(results)

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if self._period > 0 and next_iter % self._period == 0:
            if next_iter != self.trainer.max_iter:
                self._do_inspect()
