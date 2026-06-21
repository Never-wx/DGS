# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union, List
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner
from mmengine.config import ConfigDict
from mmdet.registry import HOOKS
from .transform_builder import WEIGHT_TRANSFORMS

@HOOKS.register_module()
class WeightsTransformHook(Hook):
    """Refactored WeightsTransformHook using a registry-based pattern.
    
    Args:
        cfg (dict | list[dict], optional): The configuration for weight 
            transformations. Each dict should contain a 'type' field corresponding 
            to a registered WEIGHT_TRANSFORMS. Defaults to None.
    """
    def __init__(self, cfg: Optional[Union[dict, List[dict]]] = None):
        if cfg is None:
            cfgs = [dict(type='none')] # No-op if not specified, though usually cfg is provided
        elif isinstance(cfg, (list, tuple)):
            cfgs = cfg
        else:
            cfgs = [cfg]
            
        self.transforms = []
        for c in cfgs:
            if c.get('type') == 'none':
                continue
            self.transforms.append(WEIGHT_TRANSFORMS.build(c))

    def after_load_checkpoint(self, runner: Runner, checkpoint: dict) -> None:
        """Apply transformations after loading checkpoint.
        
        Args:
            runner (Runner): The runner of the training/testing process.
            checkpoint (dict): The loaded checkpoint.
        """
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        for transform in self.transforms:
            state_dict = transform.transform(model, state_dict, work_dir=str(runner.work_dir))
            
    def before_save_checkpoint(self, runner: Runner, checkpoint: dict) -> None:
        """Apply transformations before saving checkpoint.
        
        Args:
            runner (Runner): The runner of the training/testing process.
            checkpoint (dict): The checkpoint to be saved.
        """
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        for transform in self.transforms:
            state_dict = transform.before_save(model, state_dict, work_dir=str(runner.work_dir))
