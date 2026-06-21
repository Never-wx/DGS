# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import copy
import os
import gc
from typing import Optional
from collections import OrderedDict
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner
from mmdet.registry import HOOKS
from mmengine.dist import master_only
from .merge_utils import reparameter
from mmengine.logging import MMLogger

""" 
('before_run', 'after_load_checkpoint', 'before_train',
'before_train_epoch', 'before_train_iter', 'after_train_iter',
'after_train_epoch', 'before_val', 'before_val_epoch',
'before_val_iter', 'after_val_iter', 'after_val_epoch',
'after_val', 'before_save_checkpoint', 'after_train',
'before_test', 'before_test_epoch', 'before_test_iter',
'after_test_iter', 'after_test_epoch', 'after_test', 'after_run') 
"""

@HOOKS.register_module()
class MergeHook(Hook):
    def __init__(self, cfg: dict = None):
        self.cfg = cfg if cfg else dict(type='none')

    def after_train_epoch(self, runner: Runner) -> None:
        model = runner.model
        if is_model_wrapper(model):
            model = model.module 
        
        if runner.epoch + 1 == runner.max_epochs:
            if (self.cfg.get('type') in ['rdb', 'ema']):
                reparameter(model, delete=True)
            