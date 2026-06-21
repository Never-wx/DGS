import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base_moe import BaseMoE
from ..modules.group_lora import LoraLinearRaw
from typing import Optional
from mmcv.cnn import Linear

class MoELora(BaseMoE):
    """
    A mixed expert module containing shared experts with LoRA adaptation.
    
    Args:
        in_features (int): Input feature dimension
        out_features (int): Output feature dimension
        r (int, optional): LoRA rank. Defaults to 8
        alpha (float, optional): LoRA alpha. Defaults to 16.0
        dropout (float, optional): Dropout rate. Defaults to 0.0
        merge_weights (bool, optional): Whether to merge weights. Defaults to True
        **kwargs: Additional arguments passed to BaseMOE
    """
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 r: int = 8,
                 alpha: float = 16.0,
                 dropout: float = 0.0,
                 merge_weights: bool = True,
                 **kwargs):
        
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = alpha
        self.lora_dropout = dropout
        self.merge_weights = merge_weights
        self.merged = False
        super().__init__(**kwargs)
        self.base_layer = Linear(in_features, out_features)
        # Initialize activation tracking
        self.register_buffer("activate_rate", torch.zeros([self.experts_num]), persistent=False)
        self.register_buffer("frozen_flags", torch.zeros([self.experts_num]), persistent=True)
        self.activate_count_flag = False
        self.reset_parameters()
        self.__freeze_params__()
    
    def init_experts(self):
        """Initialize expert modules with LoRA"""
        experts = nn.ModuleList()
        for _ in range(self.experts_num):
            expert = LoraLinearRaw(
                in_features=self.in_features,
                out_features=self.out_features,
                r=self.r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout
            )
            experts.append(expert)
        return experts

    def init_gates(self):
        """Initialize gating parameters"""
        gates = nn.ParameterList()
        for _ in range(self.num_tasks):
            gate = nn.Parameter(
                torch.zeros(self.experts_num, self.in_features),
                requires_grad=True
            )
            gates.append(gate)
        return gates
    
    def expert_forward(self, 
                      x: torch.Tensor,
                      task_id: int,
                      expert_idx: int,
                      add_residual: bool = False) -> torch.Tensor:
        """ Forward pass through a single expert """
        return self.experts[expert_idx](x, add_residual=add_residual)

    def __freeze_params__(self):
        """Freeze parameters based on task and activation status"""
        for param in self.base_layer.parameters():
            param.requires_grad = False   

        for i in range(self.num_tasks):
            if i != self.task_id:
                self.gates[i].requires_grad = False    

    def reset_parameters(self):
        if hasattr(self, 'gates'):
            for param in self.gates:
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))

    def forward(self, 
                hidden_states: torch.Tensor,
                add_residual: bool = False) -> torch.Tensor:
        """
        Forward pass through the MoE layer.
        
        Args:
            hidden_states (torch.Tensor): Input tensor
            add_residual (bool, optional): Whether to add residual connection. Defaults to False
            
        Returns:
            torch.Tensor: Output tensor
        """
        output = self.base_layer(hidden_states)
        moe_output, load = self.moe_forward(hidden_states, add_residual)
         
        # Update activation rates if tracking is enabled
        if self.activate_count_flag:
            self.activate_rate = self.activate_rate + load
            
        return output + moe_output
    
    # 辅助方法
    def __set_activate_count_flags__(self):
        """Enable activation rate tracking"""
        self.activate_count_flag = True

    def __set_frozen_flags__(self):
        """Set frozen flags based on activation rates"""
        k_act_nums, k_act_id = torch.topk(self.activate_rate, k=self.top_k)
        self.frozen_flags[k_act_id] = 1


