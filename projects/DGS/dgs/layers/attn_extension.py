import math
import warnings
from typing import Optional
import torch
import torch.nn as nn
import mmengine
from mmengine.utils import deprecated_api_warning
from mmcv.cnn import Linear
from mmdet.utils import OptConfigType
import mmcv.cnn.bricks.transformer as transformer
from mmcv.ops import MultiScaleDeformableAttention
from mmdet.models.utils.vlfuse_helper import BiMultiHeadAttention, BiAttentionBlock, SingleScaleBiAttentionBlock
from .modules.lora import lora_Linear
from .builder import build_peft_layer

MAX_CLAMP_VALUE = 50000

class BiMultiHeadAttention_lora(BiMultiHeadAttention):
    def __init__(self,
                 v_dim: int,
                 l_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 lora_cfg: OptConfigType = None):
        super().__init__(v_dim=v_dim, l_dim=l_dim, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        if lora_cfg is not None:
            lora_r =  lora_cfg.lora_r 
            lora_alpha = lora_cfg.lora_alpha
            lora_dropout = lora_cfg.lora_dropout
            enable_lora = lora_cfg.enable_lora

            
            self.v_proj = lora_Linear(self.v_dim, self.embed_dim, lora_r, lora_alpha, lora_dropout)
            self.l_proj = lora_Linear(self.l_dim, self.embed_dim, lora_r, lora_alpha, lora_dropout)
            self.values_v_proj = lora_Linear(self.v_dim, self.embed_dim, lora_r, lora_alpha, lora_dropout)
            self.values_l_proj = lora_Linear(self.l_dim, self.embed_dim, lora_r, lora_alpha, lora_dropout)
            if enable_lora[3]:
                self.out_v_proj = lora_Linear(self.embed_dim, self.v_dim, lora_r, lora_alpha, lora_dropout)
                self.out_l_proj = lora_Linear(self.embed_dim, self.l_dim, lora_r, lora_alpha, lora_dropout)           

class SingleScaleBiAttentionBlock_lora(BiAttentionBlock):
    def __init__(self,
                 v_dim: int,
                 l_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 drop_path: float = .0,
                 init_values: float = 1e-4,
                 lora_cfg: OptConfigType = None):
        super().__init__(v_dim=v_dim, l_dim=l_dim, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout,
                         drop_path=drop_path, init_values=init_values)
        self.attn = BiMultiHeadAttention_lora(
            v_dim=v_dim,
            l_dim=l_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            lora_cfg=lora_cfg)
        
    def forward(self, visual_feature, lang_feature, attention_mask_v=None, attention_mask_l=None):
        new_v, new_lang_feature = self.single_attention_call(
            visual_feature,
            lang_feature,
            attention_mask_v=attention_mask_v,
            attention_mask_l=attention_mask_l)
        return new_v, new_lang_feature




class BiMultiHeadAttention_MoE(BiMultiHeadAttention):
    """Refactored BiMultiHeadAttention with MoE support using the PEFT registry."""
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, dropout, moe_cfg: OptConfigType = None):
        super().__init__(v_dim=v_dim, l_dim=l_dim, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

        if moe_cfg is not None:
            self.v_proj = build_peft_layer(moe_cfg, in_features=self.v_dim, out_features=self.embed_dim)
            self.l_proj = build_peft_layer(moe_cfg, in_features=self.l_dim, out_features=self.embed_dim)
            self.values_v_proj = build_peft_layer(moe_cfg, in_features=self.v_dim, out_features=self.embed_dim)
            self.values_l_proj = build_peft_layer(moe_cfg, in_features=self.l_dim, out_features=self.embed_dim)

class SingleScaleBiAttentionBlock_MoE(SingleScaleBiAttentionBlock):
    """Refactored SingleScaleBiAttentionBlock with MoE support using the PEFT registry."""
    def __init__(self, 
                 v_dim: int,
                 l_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 drop_path: float = .0,
                 init_values: float = 1e-4,
                 moe_cfg: OptConfigType = None):
        
        super().__init__(v_dim=v_dim, l_dim=l_dim, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout,
                         drop_path=drop_path, init_values=init_values)   
        self.attn = BiMultiHeadAttention_MoE(
            v_dim=v_dim,
            l_dim=l_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            moe_cfg=moe_cfg)
