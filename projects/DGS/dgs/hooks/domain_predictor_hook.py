# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.distributed as dist
import os
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner
from mmengine.runner.checkpoint import _load_checkpoint
from mmengine.runner import load_checkpoint, load_state_dict
from mmengine import Config
from mmdet.utils import ConfigType, OptConfigType
from mmdet.registry import HOOKS
from mmengine.hooks.hook import DATA_BATCH
""" 
('before_run', 'after_load_checkpoint', 'before_train',
'before_train_epoch', 'before_train_iter', 'after_train_iter',
'after_train_epoch', 'before_val', 'before_val_epoch',
'before_val_iter', 'after_val_iter', 'after_val_epoch',
'after_val', 'before_save_checkpoint', 'after_train',
'before_test', 'before_test_epoch', 'before_test_iter',
'after_test_iter', 'after_test_epoch', 'after_test', 'after_run') 
"""

def gather_img_emebedding(feat_path, max_sample=None):
    """
    Gather image embeddings from .pt files in the given directory.
    
    Args:
        feat_path (str): Path to directory containing .pt feature files
        max_sample (int, optional): Maximum number of samples to load
        
    Returns:
        torch.Tensor: Stacked features from all files
    """
    # Get all .pt files in directory
    imgfeat_files = sorted([f for f in os.listdir(feat_path) if f.endswith('.pt')])
    
    # Load all features into a list
    feat_list = []
    for filename in imgfeat_files:
        filepath = os.path.join(feat_path, filename)
        # Load directly as torch tensor
        feat = torch.load(filepath)
        feat_list.append(feat)
    
    # Stack all features into single tensor
    features = torch.stack(feat_list, dim=0)
    
    # Apply max_sample if specified
    if max_sample is not None and len(features) > max_sample:
        rand_idx = torch.randperm(len(features))[:max_sample]
        features = features[rand_idx]
    
    return features

@HOOKS.register_module()
class DomainPredictorHooK(Hook):

    def __init__(self, cfg: OptConfigType = None) -> None:
        self.cfg = cfg if cfg else Config._dict_to_config_dict_lazy(dict(type='none'))

    def status_update(self, logger, model):
        if hasattr(model, 'domain_predictor'):
            if dist.is_available() and dist.is_initialized(): 
                dist.all_reduce(model.domain_predictor.accurate_pred, op=dist.ReduceOp.SUM)
                dist.all_reduce(model.domain_predictor.num_samples, op=dist.ReduceOp.SUM)
                dist.all_reduce(model.domain_predictor.distance, op=dist.ReduceOp.SUM)
                dist.all_reduce(model.domain_predictor.max_distance, op=dist.ReduceOp.MAX)
                dist.all_reduce(model.domain_predictor.min_distance, op=dist.ReduceOp.MIN)
                dist.barrier()
                is_main = dist.get_rank() == 0
            else:
                is_main = True
            
            if is_main:
                task_pred_acc = model.domain_predictor.accurate_pred / model.domain_predictor.num_samples
                mean_distance = model.domain_predictor.distance / model.domain_predictor.num_samples
                logger.info(f'Task prediction acc:{task_pred_acc}')
                logger.info(f'Avg distance:{mean_distance}')
                logger.info(f'Max distance:{model.domain_predictor.max_distance}')
                logger.info(f'Min distance:{model.domain_predictor.min_distance}')
            
            model.domain_predictor.accurate_pred = torch.zeros_like(model.domain_predictor.accurate_pred)
            model.domain_predictor.num_samples = torch.zeros_like(model.domain_predictor.num_samples)
            model.domain_predictor.distance = torch.zeros_like(model.domain_predictor.distance)
            model.domain_predictor.max_distance = torch.zeros_like(model.domain_predictor.max_distance)
            model.domain_predictor.min_distance = torch.ones_like(model.domain_predictor.min_distance) * 1000

    def after_val_epoch(self, runner: Runner, metrics=None) -> None:
        model = runner.model
        logger = runner.logger
        if is_model_wrapper(model):
            model = model.module  
        self.status_update(logger, model)

    def after_test_epoch(self, runner: Runner, metrics=None) -> None:
        model = runner.model
        logger = runner.logger
        if is_model_wrapper(model):
            model = model.module  
        self.status_update(logger, model)
    
    # def stop_training(self, logger, model):
    #     model.domain_predictor.early_stop()   
    #     logger('stop training domain predictor')

    # def after_train_epoch(self, runner: Runner, metrics=None):
    #     model = runner.model
    #     logger = runner.logger
    #     if is_model_wrapper(model):
    #         model = model.module  

    #     if runner.epoch + 1 == model.domain_predictor.max_epochs:
    #         self.stop_training(logger, model)

    # def after_val_epoch(self, runner: Runner, metrics=None) -> None:
    #     model = runner.model
    #     logger = runner.logger
    #     if is_model_wrapper(model):
    #         model = model.module  
    #     self.status_update(logger, model)