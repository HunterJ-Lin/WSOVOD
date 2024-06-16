#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import (default_argument_parser, default_setup,hooks,launch)
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import verify_results
from wsovod.config import add_wsovod_config


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_wsovod_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="wsovod")
    return cfg


def main(args):
    cfg = setup(args)
    if "MixedDatasets" in args.config_file:
        from wsovod.engine import DefaultTrainer_WSOVOD_MixedDatasets as Trainer_
    else:
        from wsovod.engine import DefaultTrainer_WSOVOD as Trainer_
    from detectron2.evaluation import DatasetEvaluators
    from wsovod.evaluation.ov_coco_evaluation import OVCOCOEvaluator
    
    class Trainer(Trainer_):
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

            evaluator_list.append(OVCOCOEvaluator(dataset_name, output_dir=output_folder))
            return DatasetEvaluators(evaluator_list)
    
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = {}
        res = Trainer.test_WSL(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA_WSL(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        # trainer.register_hooks([hooks.EvalHook(0, lambda: trainer.test_WSL(cfg, trainer.model))])
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA_WSL(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
