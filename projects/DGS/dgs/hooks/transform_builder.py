import torch
import torch.nn as nn
from mmengine.registry import Registry
from mmengine.logging import MMLogger
from mmengine.runner.checkpoint import _load_checkpoint
from mmengine.model import is_model_wrapper
from typing import Dict, List, Optional, Union
import copy

WEIGHT_TRANSFORMS = Registry('weight_transforms')

class BaseWeightTransform:
    def __init__(self, **kwargs):
        self.cfg = kwargs
        self.logger = MMLogger.get_current_instance()

    def transform(self, model: nn.Module, checkpoint: Dict, **kwargs) -> Dict:
        """Apply transformation after loading checkpoint."""
        return checkpoint

    def before_save(self, model: nn.Module, checkpoint: Dict, **kwargs) -> Dict:
        """Apply transformation before saving checkpoint."""
        return checkpoint

@WEIGHT_TRANSFORMS.register_module(name='moe_lora')
class MoeLoraTransform(BaseWeightTransform):
    def transform(self, model, checkpoint, **kwargs):
        save_mode = self.cfg.get('save_mode')
        if save_mode and (save_mode.get('type') == 'lora_only'):
            has_base_weights = any('language_model' in k for k in checkpoint.keys())
            
            if not has_base_weights:
                base_model_ckpt_path = save_mode.get('base_model_ckpt_path')
                self.logger.info(f"[WeightsTransform] LoRA-only checkpoint detected. Merging base weights from {base_model_ckpt_path}...")
                
                base_ckpt = _load_checkpoint(base_model_ckpt_path, map_location='cpu')
                if 'state_dict' in base_ckpt:
                    base_ckpt = base_ckpt['state_dict']
                
                for k, v in base_ckpt.items():
                    if k not in checkpoint:
                        checkpoint[k] = v
            else:
                self.logger.info(f"[WeightsTransform] Full model checkpoint detected (contains backbone). Skipping base weight merge.")

        base_layer_paths = set()
        for model_key in model.state_dict().keys():
            if '.base_layer.' in model_key:
                original_key = model_key.replace('.base_layer.', '.')
                base_layer_paths.add((original_key, model_key))
        
        for src_key, dst_key in base_layer_paths:
            if src_key in checkpoint:
                checkpoint[dst_key] = checkpoint.pop(src_key)
        return checkpoint

    def before_save(self, model, checkpoint, **kwargs):
        save_mode = self.cfg.get('save_mode')
        if save_mode and (save_mode.get('type') == 'lora_only'):
            keywords = save_mode.get('keywords')
            if not keywords:
                return checkpoint

            self.logger.info(f"[WeightsTransform] Saving lora weights only")
            for k in list(checkpoint.keys()):
                if not any(keyword in k for keyword in keywords):
                    checkpoint.pop(k)
        return checkpoint

@WEIGHT_TRANSFORMS.register_module(name='moe_group_init')
class MoeGroupInitTransform(BaseWeightTransform):
    def transform(self, model, checkpoint, **kwargs):
        if hasattr(model, 'moe_cfg') and model.moe_cfg.get('calibration', False):
            self.logger.info('[WeightsTransform] Skipping group initialization for calibration stage')
            return checkpoint
        
        task_id = getattr(model, 'task_id', 0)
        domain_predictor = getattr(model, 'domain_predictor', None)
        if domain_predictor is None:
            return checkpoint
            
        task_id_mapping = domain_predictor.task_id_mapping
        group2experts = {}  # {group_id: [task_ids]}
        for k, v in task_id_mapping.items():
            if v not in group2experts:
                group2experts[v] = []
            group2experts[v].append(k)
            
        group_id = task_id_mapping.get(task_id)
        if group_id is None:
            return checkpoint
            
        group_experts = group2experts[group_id]
        if len(group_experts) > 1:
            src_idx = group_experts[0]
            tgt_idx = group_experts[-1]

            keywords = self.cfg.get('keywords')
            if keywords is None:
                if hasattr(model, 'moe_cfg') and 'adapter' in model.moe_cfg.type:
                    keywords = ['down_proj', 'up_proj']  
                else:
                    keywords = ['lora_A', 'lora_B']
            
            for key in list(checkpoint.keys()):
                for kw in keywords:
                    # Match pattern more robustly (e.g., .kw.src_idx.weight or .kw.src_idx)
                    pattern = f'.{kw}.{src_idx}.'
                    if pattern in key:
                        new_key = key.replace(pattern, f'.{kw}.{tgt_idx}.')
                        checkpoint[new_key] = checkpoint[key].clone()
                    elif key.endswith(f'.{kw}.{src_idx}'):
                        new_key = key.replace(f'.{kw}.{src_idx}', f'.{kw}.{tgt_idx}')
                        checkpoint[new_key] = checkpoint[key].clone()
            
            # Init new gate from the base one in the group
            if 'gates' in keywords:
                for key in list(checkpoint.keys()):
                    pattern = f'gates.{group_id}'
                    if key.endswith(pattern):
                        experts_num = model.moe_cfg.get('experts_num', 1)
                        old_gate_weights = checkpoint[key]
                        old_out, in_features = old_gate_weights.shape
                        new_out = old_out + experts_num
                        new_gate_weights = old_gate_weights.new_empty(new_out, in_features)
                        new_gate_weights[:old_out] = old_gate_weights
                        new_gate_weights[old_out:] = new_gate_weights[:experts_num]
                        checkpoint[key] = new_gate_weights.clone()
        return checkpoint

