# The code is based on https://github.com/deepseek-ai/DeepSeek-MoE

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Literal
from abc import ABCMeta, abstractmethod
from .routers import MoERouter

class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss, 
    which includes the gradient of the aux loss during backpropagation.
    """
    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss

class BaseMoE(nn.Module):
    """Base class for Mixture of Experts implementations"""
    
    def __init__(self, 
                 num_tasks: int,
                 task_id: int,
                 experts_num: int,
                 top_k: int,
                 n_shared_experts: Optional[int] = None,
                 scoring_func: str = 'softmax',
                 aux_loss_alpha: float = 0.00,  # if applied, default to 0.01
                 seq_aux: bool = False,
                 norm_topk_prob: bool = True,
                 input_transform: str = 'none',
                 **kwargs):
        super().__init__()
        
        # Task configs
        self.num_tasks = num_tasks
        self.task_id = task_id if task_id is not None else num_tasks - 1
        self.experts_num = experts_num
        self.top_k = top_k
        self.n_shared_experts = n_shared_experts
        
        # Router configs
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.input_transform = input_transform
        
        # Initialize MoE components
        self.experts = self.init_experts()
        self.gates = self.init_gates()
        self.router = self.init_router()   
        
        if self.n_shared_experts is not None:
            self.shared_experts = self.init_shared_experts()
        else:
            self.shared_experts = None
            
        # Runtime state
        self.val_task_id = None
        self.spatial_shapes = None
    
    def init_router(self) -> nn.Module:
        """Initialize routers"""
        router = MoERouter(
            experts_num=self.experts_num * self.num_tasks,
            top_k=self.top_k,
            scoring_func=self.scoring_func,
            aux_loss_alpha=self.aux_loss_alpha,
            seq_aux=self.seq_aux,
            norm_topk_prob=self.norm_topk_prob,
            input_transform=self.input_transform
        )
        return router
    
    @abstractmethod
    def init_experts(self) -> nn.ModuleList:
        """Initialize expert modules"""
        raise NotImplementedError

    def init_shared_experts(self) -> Optional[nn.Module]:
        """Initialize shared experts if needed"""
        raise NotImplementedError
    
    @abstractmethod 
    def init_gates(self) -> nn.ParameterList:
        """Initialize training gate as parameter"""
        raise NotImplementedError
    
    def reset_parameters(self):
        """Reset parameters"""
        raise NotImplementedError
    
    def __freeze_params__(self):
        """Freeze parameters rules"""
        raise NotImplementedError        
    
    @abstractmethod
    def expert_forward(self, x: torch.Tensor,
                      task_id: int,
                      expert_idx: int,
                      add_residual: bool = False) -> torch.Tensor:
        """Forward pass through a single expert"""
        raise NotImplementedError
        
    def get_active_gates(self, task_id: int) -> torch.Tensor:
        """Get gates for current task"""
        return self.gates[task_id]

    def set_eval_task_id(self, task_id: int):
        """Set task ID for evaluation"""
        self.val_task_id = task_id
        
    def set_spatial_shapes(self, spatial_shapes: torch.Tensor):
        """Set spatial shapes for routing"""
        self.spatial_shapes = spatial_shapes
        
    def moe_forward(self, hidden_states: torch.Tensor,
                   add_residual: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """MoE forward pass implementation"""
        # Store original identity for residual
        identity = hidden_states
        
        # Get output shape
        orig_shape = [hidden_states.shape[0],
                     hidden_states.shape[1],
                     self.out_features]
        
        # Determine active task ID
        if self.training:
            assert self.task_id > -1, "task_id must be set during training"
            task_id = self.task_id
        
        else:
            task_id = self.val_task_id 
            if task_id == -1:  # zero-shot
                return (hidden_states.new_zeros(orig_shape),
                        hidden_states.new_zeros(self.experts_num))
                        
        # Get router outputs
        topk_idx, topk_weight, aux_loss, load = self.router(
            hidden_states,
            self.get_active_gates(task_id),
            self.spatial_shapes
        )
        
        # Process through experts
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        if self.training:
            output = self._train_forward(
                hidden_states, topk_idx, topk_weight, task_id, add_residual)
            
            if self.aux_loss_alpha > 0:
                output = AddAuxiliaryLoss.apply(output, aux_loss)
        else:
            output = self._eval_forward(
                hidden_states, topk_idx, topk_weight, task_id, add_residual)
            
        # Reshape output
        output = output.view(*orig_shape)
        
        return output, load
        
    def _train_forward(self, hidden_states: torch.Tensor,
                      topk_idx: torch.Tensor,
                      topk_weight: torch.Tensor,
                      task_id: int,
                      add_residual: bool) -> torch.Tensor:
        """Forward pass implementation for training"""
        # Repeat inputs for top-k experts
        hidden_states = hidden_states.repeat_interleave(self.top_k, dim=0)
        
        # Prepare output tensor
        output = hidden_states.new_empty([hidden_states.shape[0], self.out_features])
        
        # Process through experts
        flat_topk_idx = topk_idx.view(-1)
        for i in range(self.experts_num):
            mask = (flat_topk_idx == i)
            # if mask.any():
            expert_output = self.expert_forward(hidden_states[mask], task_id, i, add_residual)
            output[mask] = expert_output  
        
        # Combine expert outputs
        output = (output.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
               
        return output
        
    @torch.no_grad()
    def _eval_forward(self, hidden_states: torch.Tensor,
                     topk_idx: torch.Tensor,
                     topk_weight: torch.Tensor,
                     task_id: int, 
                     add_residual: bool) -> torch.Tensor:
        """Forward pass implementation for evaluation"""
        expert_cache = hidden_states.new_zeros(
            [hidden_states.shape[0], self.out_features])
            
        # Sort tokens by expert
        idxs = topk_idx.view(-1).argsort()
        tokens_per_expert = topk_idx.view(-1).bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.top_k
        
        # Process through experts
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            if start_idx == end_idx:
                continue
                
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = hidden_states[exp_token_idx]
            
            expert_out = self.expert_forward(expert_tokens, task_id, i, add_residual)
            expert_out.mul_(topk_weight.view(-1, 1)[idxs[start_idx:end_idx]])
            
            expert_cache.scatter_add_(
                0,
                exp_token_idx.view(-1, 1).repeat(1, self.out_features),
                expert_out
            )
            
        return expert_cache
    
    @abstractmethod    
    def forward(self, hidden_states: torch.Tensor,
                add_residual: bool = False) -> torch.Tensor:
        """Forward pass (to be implemented by subclasses)"""
        raise NotImplementedError
