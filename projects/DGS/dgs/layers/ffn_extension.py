import torch
import torch.nn as nn
from mmengine.model import BaseModule, Sequential
from mmengine.registry import MODELS
from mmcv.cnn import (Linear, build_activation_layer)
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.scale import LayerScale
from mmdet.utils import OptConfigType
import mmcv.cnn.bricks.transformer as transformer
from mmengine.utils import deprecated_api_warning

from .builder import build_peft_layer
from .modules.adapter import Adapter

class FFN_Adapter(transformer.FFN):
    """Implements feed-forward networks (FFNs) with identity connection with Adapter support."""
    
    def __init__(self, adapter_cfg: OptConfigType = None, **kwargs):
        super().__init__(**kwargs)
        if adapter_cfg is not None:
            self.adapter_opt = adapter_cfg.adapter_opt
            adapter_dim = adapter_cfg.adapter_dim
            adapter_drop_out = adapter_cfg.adapter_dropout
            adapter_scale= adapter_cfg.adapter_scale
            self.adapter = Adapter(d_model=self.embed_dims, bottleneck=adapter_dim, dropout=adapter_drop_out,
                                adapter_scalar=adapter_scale)
        else:
            self.adapter_opt = None 
    
    def forward(self, x, identity=None):
        if identity is None:
            identity = x

        out = self.layers(x)
        out = self.gamma2(out)

        if not self.add_identity:
            return self.dropout_layer(out)
        else:
            out = self.dropout_layer(out)
        
        if self.adapter_opt == 'parallel':
            adapt_x = self.adapter(x, add_residual=False)  
            out = out + adapt_x                         

        elif self.adapter_opt == 'sequential':
            out = self.adapter(out, add_residual=True)
            
        return identity + out

    def mark_only_adapter_trainable(self):
        pass

    def merge_adapter_weights(self):
        pass

class FFN_MoE(BaseModule):
    """Refactored FFN with MoE support using the PEFT registry."""

    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=1024,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU', inplace=True),
                 ffn_drop=0.,
                 dropout_layer=None,
                 add_identity=True,
                 init_cfg=None,
                 layer_scale_init_value=0.,
                 moe_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        assert num_fcs >= 2, f'num_fcs should be no less than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.ffn_drop = ffn_drop
        self.moe_cfg = moe_cfg
        self.act_cfg = act_cfg

        self._init_layers()
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity

        if layer_scale_init_value > 0:
            self.gamma2 = LayerScale(embed_dims, scale=layer_scale_init_value)
        else:
            self.gamma2 = nn.Identity()
    
    def _init_layers(self) -> None:
        if self.moe_cfg is None:
            self.layers = self._build_standard_ffn()
        elif 'adapter' in self.moe_cfg['type'].lower():
            self.layers = self._build_standard_ffn()
            self.moe = build_peft_layer(self.moe_cfg, in_features=self.embed_dims, out_features=self.embed_dims)
        else:
            self.layers = self._build_peft_ffn()

    def _build_standard_ffn(self) -> nn.Sequential:
        layers = []
        for _ in range(self.num_fcs - 1):
            fc_layer = Linear(self.embed_dims, self.feedforward_channels)
            act_layer = build_activation_layer(self.act_cfg)
            dropout = nn.Dropout(self.ffn_drop)
            layers.append(Sequential(fc_layer, act_layer, dropout))
        
        out_layer = Linear(self.feedforward_channels, self.embed_dims)
        layers.append(out_layer)
        layers.append(nn.Dropout(self.ffn_drop))
        return Sequential(*layers)
    
    def _build_peft_ffn(self) -> nn.Sequential:
        layers = []
        for _ in range(self.num_fcs - 1):
            fc_layer = build_peft_layer(
                self.moe_cfg,
                in_features=self.embed_dims, 
                out_features=self.feedforward_channels)
            act_layer = build_activation_layer(self.act_cfg)
            dropout = nn.Dropout(self.ffn_drop)
            layers.append(Sequential(fc_layer, act_layer, dropout))
        
        out_layer = build_peft_layer(
            self.moe_cfg,
            in_features=self.feedforward_channels, 
            out_features=self.embed_dims)
        
        layers.append(out_layer)
        layers.append(nn.Dropout(self.ffn_drop))
        return Sequential(*layers)

    def forward(self, x, identity=None):
        if identity is None:
            identity = x

        out = self.layers(x)
        out = self.gamma2(out)
        out = self.dropout_layer(out)

        if self.moe_cfg is not None and hasattr(self, 'moe'):
            adapt_x = self.moe(x, add_residual=False)
            out = out + adapt_x
        
        if not self.add_identity:
            return out
        else:
            return identity + out
