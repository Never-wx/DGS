# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks.hook import DATA_BATCH
import torch.nn as nn
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner
from mmengine import Config
from mmdet.utils import ConfigType, OptConfigType
from mmdet.registry import HOOKS
import torch
import os
from mmengine.dist import master_only
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
class MOEHook(Hook):

    def __init__(self, cfg: OptConfigType = None):
        self.cfg = cfg if cfg else Config._dict_to_config_dict_lazy(dict(type='none'))
        self.type = self.cfg.type

    @master_only
    def output_topk_activate_list(self, work_dir, model):
        work_dir = os.path.dirname(work_dir)
        output_path = os.path.join(work_dir, 'frozen_experts.txt')
        with open(output_path, 'a') as f:
            for name, module in model.named_modules():
                if hasattr(module, 'frozen_flags'):
                    # 获取被冻结的专家索引
                    frozen_indices = torch.nonzero(module.frozen_flags).squeeze(-1).tolist()
                    if not isinstance(frozen_indices, list):
                        frozen_indices = [frozen_indices]
                    
                    # 获取一个专家的所有参数名称模板
                    param_names = [name for name, _ in module.experts[0].named_parameters()]
                    
                    # 只为被冻结的专家生成完整参数路径
                    for idx in frozen_indices:
                        for param_name in param_names:
                            full_param_name = f"{name}.experts.{idx}.{param_name}"
                            f.write(f"{full_param_name}\n")

    def set_flags(self, model, cur_epoch, max_epochs, work_dir, rank):
        """Output list of frozen_flags""" 
        if cur_epoch + 2 == max_epochs:   # ep9
            for module in model.modules():
                if hasattr(module, '__set_activate_count_flags__'):     
                    module.__set_activate_count_flags__()   
                
        if cur_epoch + 1 == max_epochs:   # ep10
            for module in model.modules():
                if hasattr(module, 'activate_rate'):  
                    if torch.distributed.is_initialized():
                        torch.distributed.all_reduce(module.activate_rate, op=torch.distributed.ReduceOp.SUM)
                        module.activate_rate /= torch.distributed.get_world_size()
                        module.__set_frozen_flags__()
            self.output_topk_activate_list(work_dir, model)
    
    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        model = runner.model
        if is_model_wrapper(model):
            model = model.module   

    def before_train(self, runner) -> None:
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

    def after_train_epoch(self, runner: Runner) -> None:
        model = runner.model
        if is_model_wrapper(model):
            model = model.module   
        if self.type == 'freeze_topk_activate':
            self.set_flags(model, runner.epoch, runner.max_epochs, runner.work_dir, runner.rank)
