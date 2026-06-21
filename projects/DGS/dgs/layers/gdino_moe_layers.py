# Copyright (c) OpenMMLab. All rights reserved.
import torch
import os
import numpy as np
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention
from mmengine.model import ModuleList
from mmengine.model import ModuleList
from mmdet.models.layers.transformer.deformable_detr_layers import DeformableDetrTransformerEncoderLayer
from mmdet.models.layers.transformer.detr_layers import DetrTransformerEncoderLayer
from mmdet.models.layers.transformer.dino_layers import DinoTransformerDecoder
from mmdet.models.layers.transformer.grounding_dino_layers import (GroundingDinoTransformerEncoder, GroundingDinoTransformerDecoder,
                                                                   GroundingDinoTransformerDecoderLayer)
from mmdet.models.utils.vlfuse_helper import SingleScaleBiAttentionBlock
from mmdet.models.layers.transformer.utils import MLP, get_text_sine_pos_embed
from torch import Tensor
from .ffn_extension import FFN_MoE
from .attn_extension import SingleScaleBiAttentionBlock_MoE, BiMultiHeadAttention_MoE
from mmdet.models.utils.vlfuse_helper import VLFuse, BertEncoderLayer, BiAttentionBlock
from mmdet.models.dense_heads.atss_vlfusion_head import DyReLU, DyConv

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None

class GroundingDinoTransformerDecoderLayer_MoE(GroundingDinoTransformerDecoderLayer):
    
    def __init__(self, moe_cfg, **kwargs) -> None:
        if hasattr(moe_cfg, 'dec_moe_cfg'):
            self.moe_cfg = moe_cfg.dec_moe_cfg
            self.moe_cfg.num_tasks = moe_cfg.num_tasks
            self.moe_cfg.task_id = moe_cfg.task_id
        else:
            self.moe_cfg = moe_cfg
        
        super().__init__(**kwargs)
    
    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn_text = MultiheadAttention(**self.cross_attn_text_cfg)
        self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN_MoE(moe_cfg=self.moe_cfg, **self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(4)
        ]
        self.norms = ModuleList(norms_list)

class GroundingDinoTransformerDecoder_MoE(DinoTransformerDecoder):

    def __init__(self, moe_cfg, **kwargs) -> None:
        self.moe_cfg = moe_cfg  
        self.replace_layer_type = moe_cfg.get('replace_layer_type', 'none')
        self.replace_dec_layer_ids = moe_cfg.get('replace_dec_layer_ids')
        super().__init__(**kwargs)
    
    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList()
        for i in range(self.num_layers):
            if 'dec_ffn' in self.replace_layer_type and i in self.replace_dec_layer_ids:
                    self.layers.append(GroundingDinoTransformerDecoderLayer_MoE(moe_cfg=self.moe_cfg, **self.layer_cfg))
            else:
                self.layers.append(GroundingDinoTransformerDecoderLayer(**self.layer_cfg))
        
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

class DeformableDetrTransformerEncoderLayer_MoE(DeformableDetrTransformerEncoderLayer):
    def __init__(self, moe_cfg, **kwargs) -> None:
        self.moe_cfg = moe_cfg
        super().__init__(**kwargs)
        
    def _init_layers(self) -> None:
        """Initialize self_attn, ffn, and norms."""
        self.self_attn = MultiScaleDeformableAttention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN_MoE(moe_cfg=self.moe_cfg, **self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(2)
        ]
        self.norms = ModuleList(norms_list)
    
class DetrTransformerEncoderLayer_MoE(DetrTransformerEncoderLayer):
    
    def __init__(self, moe_cfg, **kwargs) -> None:
        self.moe_cfg = moe_cfg
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN_MoE(moe_cfg=self.moe_cfg, **self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(2)
        ]
        self.norms = ModuleList(norms_list)
    
class GroundingDinoTransformerEncoder_MoE(GroundingDinoTransformerEncoder):

    def __init__(self, moe_cfg, **kwargs) -> None:
        self.moe_cfg = moe_cfg
        self.save_intermediate_feats = True if 'save_path' in self.moe_cfg else False 
        self.replace_layer_type = moe_cfg.get('replace_layer_type', 'none')
        self.replace_enc_layer_ids = moe_cfg.get('replace_enc_layer_ids')
        super().__init__(**kwargs)
    
    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList()
        self.text_layers = ModuleList()
        self.fusion_layers = ModuleList()

        for i in range(self.num_layers):
            if 'enc_ffn_img' in self.replace_layer_type and i in self.replace_enc_layer_ids:
                self.layers.append(DeformableDetrTransformerEncoderLayer_MoE(moe_cfg=self.moe_cfg, **self.layer_cfg))
            else:
                self.layers.append(DeformableDetrTransformerEncoderLayer(**self.layer_cfg)) 
            
            if 'enc_ffn_text' in self.replace_layer_type and i in self.replace_enc_layer_ids:
                self.text_layers.append(DetrTransformerEncoderLayer_MoE(moe_cfg=self.moe_cfg, **self.text_layer_cfg))
            else:
                self.text_layers.append(DetrTransformerEncoderLayer(**self.text_layer_cfg))  

            if 'enc_fusion' in self.replace_layer_type and i in self.replace_enc_layer_ids:                       
                self.fusion_layers.append(SingleScaleBiAttentionBlock_MoE(moe_cfg=self.moe_cfg, **self.fusion_layer_cfg))
            else:
                self.fusion_layers.append(SingleScaleBiAttentionBlock(**self.fusion_layer_cfg))   
        
        self.embed_dims = self.layers[0].embed_dims
        
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])
                self.fusion_layers[i] = checkpoint_wrapper(self.fusion_layers[i])
    
        






class BiAttentionBlock_MoE(BiAttentionBlock):
    def __init__(self, moe_cfg, **kwargs) -> None:
        self.moe_cfg = moe_cfg
        super().__init__(**kwargs)
        self.attn = BiMultiHeadAttention_MoE(moe_cfg=self.moe_cfg, 
                                             v_dim=kwargs['v_dim'],
                                             l_dim=kwargs['l_dim'],
                                             embed_dim=kwargs['embed_dim'],
                                             num_heads=kwargs['num_heads'],
                                             dropout=kwargs.get('dropout', 0.1))

class VLFuse_MoE(VLFuse):
    def __init__(self, moe_cfg, **kwargs) -> None:
        self.moe_cfg = moe_cfg
        super().__init__(**kwargs)
        self.b_attn = BiAttentionBlock_MoE(
            moe_cfg=self.moe_cfg,
            v_dim=256,
            l_dim=768,
            embed_dim=2048,
            num_heads=8,
            dropout=0.1,
            drop_path=0.0,
            init_values=1.0 / 6.0)

class BertEncoderLayer_MoE(BertEncoderLayer):
    def __init__(self, moe_cfg, **kwargs) -> None:
        self.moe_cfg = moe_cfg
        super().__init__(**kwargs)
        self.ffn = FFN_MoE(moe_cfg=self.moe_cfg, 
                           embed_dims=self.config.hidden_size,
                           feedforward_channels=self.config.intermediate_size,
                           num_fcs=2,
                           add_identity=True,
                           dropout_layer=dict(type='Dropout', drop_prob=self.config.hidden_dropout_prob))
    
    def feed_forward_chunk(self, attention_output: Tensor) -> Tensor:
        return self.ffn(attention_output)

class DyReLU_MoE(DyReLU):
    def __init__(self, moe_cfg, channels, **kwargs):
        self.moe_cfg = moe_cfg
        # In DyConv, in_channels and out_channels for DyReLU are usually the same (the out_channels of DyConv)
        super().__init__(in_channels=channels, out_channels=channels, **kwargs)
        
        # Replace self.fc with FFN_MoE for adaptive routing
        # DyReLU.fc is Sequential(Linear, ReLU, Linear, Hardsigmoid)
        # We replace the routing logic with MoE
        self.fc = nn.Sequential(
            FFN_MoE(moe_cfg=self.moe_cfg,
                    embed_dims=channels,
                    feedforward_channels=channels // kwargs.get('expand_ratio', 4),
                    num_fcs=2,
                    add_identity=False,
                    act_cfg=dict(type='ReLU', inplace=True)),
            nn.Hardsigmoid(inplace=True)
        )

class DyConv_MoE(DyConv):
    def __init__(self, moe_cfg, **kwargs):
        self.moe_cfg = moe_cfg
        super().__init__(**kwargs)
        
        # DyConv doesn't have _init_layers, it initializes self.relu in __init__.
        # We replace it here if use_dyrelu is True.
        if kwargs.get('use_dyrelu', False):
            out_channels = kwargs.get('out_channels')
            self.relu = DyReLU_MoE(moe_cfg=self.moe_cfg, 
                                   channels=out_channels)
