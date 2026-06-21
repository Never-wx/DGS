from mmdet.utils import OptConfigType
from .modules.lora import lora_Linear, lora_MergedLinear
from .modules.replinear import RepLinear
from .modules.adapter import Adapter

# Local DGS MoE layers
from .moe_layers.moe_lora import MoELora
from .moe_layers.noisy_moe_adapter import NoisyMoEAdapter
from .moe_layers.moe_adaptive_expand_lora import AdaptiveExpandMoELora

_TYPE_MAP = {
    'lora': lora_Linear,
    'lora_merged': lora_MergedLinear,
    'rep_linear': RepLinear,
    'adapter': Adapter,
    'moe_lora': MoELora,
    'noisy_moe_adapter': NoisyMoEAdapter,
    'moe_adaptive_expand_lora': AdaptiveExpandMoELora,
}

def build_peft_layer(cfg: OptConfigType, **kwargs):
    """
    Unified factory function for all PEFT/MoE layers in DGS.
    
    Args:
        cfg (dict or ConfigDict): Configuration for the PEFT layer, must contain 'type'.
        **kwargs: Additional arguments to override or supplement the config.
        
    Returns:
        Instantiated PEFT/MoE layer.
    """
    if cfg is None:
        return None
    
    if isinstance(cfg, dict):
        full_cfg = cfg.copy()
    else:
        full_cfg = dict(cfg)
        
    full_cfg.update(kwargs)
    
    # Extract type and fallback to 'lora' if not specified
    layer_type = full_cfg.pop('type', 'lora')
    
    if layer_type in _TYPE_MAP:
        cls = _TYPE_MAP[layer_type]
        return cls(**full_cfg)
    
    raise ValueError(f"Unknown PEFT layer type: {layer_type}. "
                     f"Supported types: {list(_TYPE_MAP.keys())}")
