# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import math
import os
import time
from collections import OrderedDict

import detectron2.utils.comm as comm
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from fvcore.nn.precise_bn import get_bn_modules
from detectron2.data import MetadataCatalog
from detectron2.evaluation import (DatasetEvaluators,
                                   LVISEvaluator, PascalVOCDetectionEvaluator)
from detectron2.modeling import GeneralizedRCNNWithTTA
from wsovod.evaluation import PascalVOCDetectionEvaluator_WSL,COCOEvaluator
from .defaults import DefaultTrainer
from wsovod.modeling import GeneralizedRCNNWithTTAAVG, GeneralizedRCNNWithTTAUNION
from wsovod.data.build_multi_dataset import build_detection_train_loader_multi_dataset

class DefaultTrainer_WSOVOD(DefaultTrainer):

    def __init__(self, cfg):
        cfg = DefaultTrainer_WSOVOD.auto_scale_workers(cfg, comm.get_world_size())
        super().__init__(cfg)

        self._data_loader_iter = iter(self.data_loader)
        self.iter_size = cfg.WSOVOD.ITER_SIZE
        self.cfg = cfg
        self.filter_empty = cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS
        self.output_dir = cfg.OUTPUT_DIR
        os.makedirs(os.path.join(self.output_dir, "vis"), exist_ok=True)

    def run_step(self):
        self._trainer.iter = self.iter
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """
        while True:
            data = next(self._data_loader_iter)
            if not self.filter_empty or all([len(x["instances"]) > 0 for x in data]):
                break

        data_time = time.perf_counter() - start

        """
        If your want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)

        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        """
        If you need accumulate gradients or something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        if self.iter == self.start_iter:
            self.optimizer.zero_grad()

        if self.iter_size > 1:
            losses = losses / self.iter_size
        losses.backward()

        self._trainer._write_metrics(loss_dict, data_time)
        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        if self.iter % self.iter_size == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_" + dataset_name)
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type in ["coco"]:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator_WSL(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger(__name__)
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA_" + name)
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    @classmethod
    def test_with_TTA_WSL(cls, cfg, model):
        if cfg.TEST.EVAL_TRAIN:
            cfg.defrost()
            DATASETS_TEST = cfg.DATASETS.TEST
            DATASETS_PROPOSAL_FILES_TEST = cfg.DATASETS.PROPOSAL_FILES_TEST
            # cfg.DATASETS.TEST = cfg.DATASETS.TEST + cfg.DATASETS.TRAIN
            # cfg.DATASETS.PROPOSAL_FILES_TEST = (
            #     cfg.DATASETS.PROPOSAL_FILES_TEST + cfg.DATASETS.PROPOSAL_FILES_TRAIN
            # )
            cfg.DATASETS.TEST = cfg.DATASETS.TRAIN + cfg.DATASETS.TEST
            cfg.DATASETS.PROPOSAL_FILES_TEST = (
                cfg.DATASETS.PROPOSAL_FILES_TRAIN + cfg.DATASETS.PROPOSAL_FILES_TEST
            )
            cfg.freeze()

        logger = logging.getLogger(__name__)
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        # if cfg.MODEL.LOAD_PROPOSALS:
        if cfg.MODEL.PROPOSAL_GENERATOR.NAME == "PrecomputedProposals":
            model = GeneralizedRCNNWithTTAAVG(cfg, model)
        else:
            model = GeneralizedRCNNWithTTAUNION(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA_" + name)
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})

        if cfg.TEST.EVAL_TRAIN:
            cfg.defrost()
            cfg.DATASETS.TEST = DATASETS_TEST
            cfg.DATASETS.PROPOSAL_FILES_TEST = DATASETS_PROPOSAL_FILES_TEST
            cfg.freeze()

        return res

    @classmethod
    def test_WSL(cls, cfg, model):
        if cfg.TEST.EVAL_TRAIN:
            cfg.defrost()
            DATASETS_TEST = cfg.DATASETS.TEST
            DATASETS_PROPOSAL_FILES_TEST = cfg.DATASETS.PROPOSAL_FILES_TEST
            # cfg.DATASETS.TEST = cfg.DATASETS.TEST + cfg.DATASETS.TRAIN
            # cfg.DATASETS.PROPOSAL_FILES_TEST = (
            #     cfg.DATASETS.PROPOSAL_FILES_TEST + cfg.DATASETS.PROPOSAL_FILES_TRAIN
            # )
            cfg.DATASETS.TEST = cfg.DATASETS.TRAIN + cfg.DATASETS.TEST
            cfg.DATASETS.PROPOSAL_FILES_TEST = (
                cfg.DATASETS.PROPOSAL_FILES_TRAIN + cfg.DATASETS.PROPOSAL_FILES_TEST
            )
            cfg.freeze()

        logger = logging.getLogger(__name__)
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference for WSL ...")
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_" + name)
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k: v for k, v in res.items()})

        if cfg.TEST.EVAL_TRAIN:
            cfg.defrost()
            cfg.DATASETS.TEST = DATASETS_TEST
            cfg.DATASETS.PROPOSAL_FILES_TEST = DATASETS_PROPOSAL_FILES_TEST
            cfg.freeze()

        return res


class DefaultTrainer_WSOVOD_MixedDatasets(DefaultTrainer):

    def __init__(self, cfg):
        cfg = DefaultTrainer_WSOVOD_MixedDatasets.auto_scale_workers(cfg, comm.get_world_size())
        super().__init__(cfg)

        self._data_loader_iter = iter(self.data_loader)
        self.iter_size = cfg.WSOVOD.ITER_SIZE
        self.cfg = cfg
        self.filter_empty = cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS
        self.output_dir = cfg.OUTPUT_DIR
        os.makedirs(os.path.join(self.output_dir, "vis"), exist_ok=True)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_train_loader_multi_dataset(cfg)

    def run_step(self):
        self._trainer.iter = self.iter
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """
        while True:
            data = next(self._data_loader_iter)
            if not self.filter_empty or all([len(x["instances"]) > 0 for x in data]):
                break

        data_time = time.perf_counter() - start

        """
        If your want to do something with the losses, you can wrap the model.
        """
        source_id = data[0]["dataset_id"]
        for d in data:
            assert source_id == d["dataset_id"]

        loss_dict = self.model(data)

        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        """
        If you need accumulate gradients or something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        if self.iter == self.start_iter:
            self.optimizer.zero_grad()

        if self.iter_size > 1:
            losses = losses / self.iter_size
        losses.backward()

        # loss_dict_t = {}
        # for key in loss_dict.keys():
        #     loss_dict_t[key + '_' +self.cfg.DATASETS.MIXED_DATASETS.NAMES[source_id]] = loss_dict[key]
        # loss_dict.update(loss_dict_t)
        # loss_dict[self.cfg.DATASETS.MIXED_DATASETS.NAMES[source_id]] = self.model.parameters().sum()*0.0
        self._trainer._write_metrics(loss_dict, data_time)
        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        if self.iter % self.iter_size == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_" + dataset_name)
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type in ["coco"]:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator_WSL(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger(__name__)
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA_" + name)
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    @classmethod
    def test_with_TTA_WSL(cls, cfg, model):
        if cfg.TEST.EVAL_TRAIN:
            cfg.defrost()
            DATASETS_TEST = cfg.DATASETS.TEST
            DATASETS_PROPOSAL_FILES_TEST = cfg.DATASETS.PROPOSAL_FILES_TEST
            # cfg.DATASETS.TEST = cfg.DATASETS.TEST + cfg.DATASETS.TRAIN
            # cfg.DATASETS.PROPOSAL_FILES_TEST = (
            #     cfg.DATASETS.PROPOSAL_FILES_TEST + cfg.DATASETS.PROPOSAL_FILES_TRAIN
            # )
            cfg.DATASETS.TEST = cfg.DATASETS.TRAIN + cfg.DATASETS.TEST
            cfg.DATASETS.PROPOSAL_FILES_TEST = (
                cfg.DATASETS.PROPOSAL_FILES_TRAIN + cfg.DATASETS.PROPOSAL_FILES_TEST
            )
            cfg.freeze()

        logger = logging.getLogger(__name__)
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        # if cfg.MODEL.LOAD_PROPOSALS:
        if cfg.MODEL.PROPOSAL_GENERATOR.NAME == "PrecomputedProposals":
            model = GeneralizedRCNNWithTTAAVG(cfg, model)
        else:
            model = GeneralizedRCNNWithTTAUNION(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA_" + name)
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})

        if cfg.TEST.EVAL_TRAIN:
            cfg.defrost()
            cfg.DATASETS.TEST = DATASETS_TEST
            cfg.DATASETS.PROPOSAL_FILES_TEST = DATASETS_PROPOSAL_FILES_TEST
            cfg.freeze()

        return res

    @classmethod
    def test_WSL(cls, cfg, model):
        if cfg.TEST.EVAL_TRAIN:
            cfg.defrost()
            DATASETS_TEST = cfg.DATASETS.TEST
            DATASETS_PROPOSAL_FILES_TEST = cfg.DATASETS.PROPOSAL_FILES_TEST
            # cfg.DATASETS.TEST = cfg.DATASETS.TEST + cfg.DATASETS.TRAIN
            # cfg.DATASETS.PROPOSAL_FILES_TEST = (
            #     cfg.DATASETS.PROPOSAL_FILES_TEST + cfg.DATASETS.PROPOSAL_FILES_TRAIN
            # )
            cfg.DATASETS.TEST = cfg.DATASETS.TRAIN + cfg.DATASETS.TEST
            cfg.DATASETS.PROPOSAL_FILES_TEST = (
                cfg.DATASETS.PROPOSAL_FILES_TRAIN + cfg.DATASETS.PROPOSAL_FILES_TEST
            )
            cfg.freeze()

        logger = logging.getLogger(__name__)
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference for WSL ...")
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_" + name)
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k: v for k, v in res.items()})

        if cfg.TEST.EVAL_TRAIN:
            cfg.defrost()
            cfg.DATASETS.TEST = DATASETS_TEST
            cfg.DATASETS.PROPOSAL_FILES_TEST = DATASETS_PROPOSAL_FILES_TEST
            cfg.freeze()

        return res
