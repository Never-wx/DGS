import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Literal
from .base_moe import BaseMoE, AddAuxiliaryLoss
from ..modules.group_lora import (LoraLinearRaw, LoraLinearShare, LoraLinearRep, LoraLinearSeperate)
from .routers import GlobalRouter

class AdaptiveExpandMoELora(BaseMoE):
    """
    A mixed expert module containing shared experts with LoRA adaptation.
    
    Args:
        in_features (int): Input feature dimension
        out_features (int): Output feature dimension
        r (int, optional): LoRA rank. Defaults to 8
        alpha (float, optional): LoRA alpha. Defaults to 16.0
        dropout (float, optional): Dropout rate. Defaults to 0.0
        **kwargs: Additional arguments passed to DS_MoE_base
    """
    def __init__(self, 
                in_features: int,
                out_features: int,
                r: int = 8,
                alpha: float = 16.0,
                dropout: float = 0.0,
                group_cfg: dict = None,
                **kwargs):
        
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.group_cfg = group_cfg
        super().__init__(**kwargs)
        # Initialize main branch parameters
        self.base_layer = nn.Linear(in_features, out_features)
        self.reset_parameters()
        self.switch_signal=False
    
    def init_router(self) -> nn.Module:
        """Initialize routers"""
        router = GlobalRouter(
            experts_num=self.experts_num,
            top_k=self.top_k,
            scoring_func=self.scoring_func,
            aux_loss_alpha=self.aux_loss_alpha,
            seq_aux=self.seq_aux,
            norm_topk_prob=self.norm_topk_prob,
            input_transform=self.input_transform
        )
        return router
    
    def init_gates(self, task_id=None) -> nn.ParameterList:  # global router w/o gates
        """Initialize gating parameters"""
        return None    

    def get_active_gates(self, task_id: int) -> torch.Tensor:
        """Get gates for current task"""
        return None

    def reset_parameters(self):
        """Reset gates parameters"""
        if self.gates is not None:
            for param in self.gates:
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))

    def init_experts(self, task2group=None, group2tasks=None) -> nn.ModuleList:
        """Initialize expert modules with LoRA"""
        if task2group is None:
            return None

        experts = nn.ModuleList()
        GroupExpertClass = {
            'raw': LoraLinearRaw,
            'share': LoraLinearShare,
            'rep': LoraLinearRep,
            'sep': LoraLinearSeperate,
        }
        
        for group_id, task_ids in group2tasks.items():
            expert = GroupExpertClass[self.group_cfg.type](
                in_features=self.in_features,
                out_features=self.out_features,
                r=self.r,
                lora_alpha=self.alpha,
                lora_dropout=self.dropout,
                task_ids=task_ids,
                cur_task_id=max(task2group.keys()),
                group_cfg=self.group_cfg,
            )
            experts.append(expert)
        return experts

    def __freeze_params__(self, task2group):
        """Freeze parameters for previous tasks"""
        cur_task_id = max(task2group.keys())
        num_groups = max(task2group.values()) + 1
        group_id = task2group[cur_task_id]
        
        # Freeze main branch
        for param in self.base_layer.parameters():
            param.requires_grad = False          
        
        # freeze all experts except for the current task
        for i in range(num_groups):
            if i != group_id:
                for param in self.experts[i].parameters():
                    param.requires_grad = False
            else:  # special case for group of current task
                self.experts[i].__freeze_params__(cur_task_id)
        
    def group_init(self, task2group, group2tasks):
        # transform task_id to group_id for training time forward
        self.task_id = task2group[self.task_id]
        
        self.experts = self.init_experts(task2group=task2group, 
                                         group2tasks=group2tasks) 
        # self.gates = self.init_gates(group_id)
        self.__freeze_params__(task2group)
    
    def set_switch_signal(self, signal):
        """Switch to base expert forward for distillation"""
        self.switch_signal = signal

    def expert_forward(self, 
                      x: torch.Tensor,
                      group_id: int,
                      expert_idx: int,
                      add_residual: bool = False) -> torch.Tensor: 

        if self.switch_signal:
            return self.experts[group_id].forward_base_expert(x, expert_idx=expert_idx)
        
        return self.experts[group_id](x, expert_idx=expert_idx)
    
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
        # Main branch output
        # Add shared experts if present
        output = self.base_layer(hidden_states)
        if self.shared_experts is not None:
            output = output + self.shared_experts(hidden_states)
        # MoE branch output
        moe_output, _ = self.moe_forward(hidden_states, add_residual)
        
        return output + moe_output

