import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List

class LoraLinearRaw(nn.Module):
    """LoRA implemented in a dense layer"""
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        **kwargs,
    ):  
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features), requires_grad=True)
            self.lora_B = nn.Parameter(torch.zeros(out_features, r), requires_grad=True)
            self.scaling = self.lora_alpha / self.r
            self.reset_parameters()
    
    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor, add_residual=None): 
        input_dtype = x.dtype   
        result = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        return result.to(input_dtype)

class LoraLinearShare(nn.Module):
    """LoRA implemented in a dense layer with shared A matrix across tasks in the same group"""
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        task_ids: List[int] = None,
        group_cfg: dict = None,
        **kwargs,
    ):  
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        self.task_ids = task_ids if task_ids is not None else []
        self.merge_method = group_cfg.get('merge_method', 'raw_concat')
        self.merged = False

        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        
        # Shared LoRA A matrix
        self.lora_A = nn.Parameter(torch.zeros(r, in_features), requires_grad=True)

        # Task-specific B matrices
        self.lora_B = nn.ParameterDict()
        for task_id in self.task_ids:
            self.lora_B[str(task_id)] = nn.Parameter(
                torch.zeros(out_features, r), requires_grad=True
            )
        
        if self.merge_method == 'learned':
            self.reweight_factor = nn.ParameterDict()
            for task_id in self.task_ids:
                self.reweight_factor[str(task_id)] = nn.Parameter(
                    torch.ones(1) * 0.5, requires_grad=True
                )
        else:
            self.reweight_factor = {} # Changed to a standard dictionary
            for task_id in self.task_ids:
                self.reweight_factor[str(task_id)] = torch.ones(1) * 0.5      

        self.reset_parameters()
    
    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        # if not self.merge_method == 'learned':
        for tid in self.task_ids:
            B = self.lora_B[str(tid)]
            self.reweight_factor[str(tid)].data = self.reweight_factor[str(tid)].data.to(B.device)     
        
        if mode:
            if self.merged:
                # Make sure that the weights are not merged
                merge_weight = 0
                for tid in self.task_ids[1:]:
                    B = self.lora_B[str(tid)]
                    weight = self.reweight_factor[str(tid)]
                    merge_weight += B * weight  

                self.lora_B[str(self.task_ids[0])].data = \
                    (self.lora_B[str(self.task_ids[0])].data - merge_weight) / self.reweight_factor[str(self.task_ids[0])]    
                self.merged = False
        else:
            if not self.merged:
                # Merge the weights and mark it
                merge_weight = 0
                for tid in self.task_ids:
                    B = self.lora_B[str(tid)]
                    weight = self.reweight_factor[str(tid)]
                    merge_weight += B * weight     
                
                self.lora_B[str(self.task_ids[0])].data = merge_weight 
                self.merged = True       
    
    def __freeze_params__(self, cur_task_id):
        if self.merge_method == 'raw_concat':   # raw_concat, freeze previous A and B[0:t-1] 
            for task_id in self.task_ids:
                if task_id != cur_task_id:
                    self.lora_B[str(task_id)].requires_grad = False

            if len(self.task_ids) > 1: # frozen A after step 1
                self.lora_A.requires_grad = False 

        elif self.merge_method == 'slowA_concat':   # freeze B[0:t-1], A update slowly in following steps
            for task_id in self.task_ids:
                if task_id != cur_task_id:
                    self.lora_B[str(task_id)].requires_grad = False
        
        elif self.merge_method == 'learned':
            pass
            
    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            # initialize A with kaiming uniform
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # initialize all B matrices to zero
            for task_id in self.task_ids:
                nn.init.zeros_(self.lora_B[str(task_id)])
    
    def merge_forward(self, x, cur_task_id=None, expert_idx=None):
        input_dtype = x.dtype

        if not self.merged:
            result = 0
            for tid in self.task_ids:
                B = self.lora_B[str(tid)]
                weight = self.reweight_factor[str(tid)]
                result += B * weight      

            result = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ result.transpose(0, 1)) * self.scaling
            
        else:
            result = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ 
                        self.lora_B[str(self.task_ids[0])].transpose(0, 1)) * self.scaling
            
        return result.to(input_dtype) 
    
    def raw_forward(self, x, cur_task_id=None, expert_idx=None): 
        input_dtype = x.dtype
        
        # If task_id not specified and in training, default to latest task
        if cur_task_id is None:
            cur_task_id = self.task_ids[-1]

        # raw forward for training
        # if self.training and not self.merged:
        #     B = self.lora_B[str(cur_task_id)]
        #     result = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ B.transpose(0, 1)) * self.scaling
        if not self.merged:
            result = 0
            for tid in self.task_ids:
                B = self.lora_B[str(tid)]
                weight = self.reweight_factor[str(tid)]
                result += B * weight      

            result = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ result.transpose(0, 1)) * self.scaling        

        else:  # For inference, average all task outputs
            result = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ 
                        self.lora_B[str(self.task_ids[0])].transpose(0, 1)) * self.scaling 
                               
        return result.to(input_dtype)
    
    def forward(self, x, cur_task_id=None, expert_idx=None): 
        if self.merge_method == 'learned':
            return self.merge_forward(x, cur_task_id, expert_idx)
        else:   
            return self.raw_forward(x, cur_task_id, expert_idx)
    
class LoraLinearSeperate(nn.Module):
    """LoRA implemented in seperate A matrix across tasks in the same group"""
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        task_ids: List[int] = None,
        group_cfg: dict = None,
        **kwargs,
    ):  
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        self.task_ids = task_ids if task_ids is not None else []
        self.merge_method = group_cfg.get('merge_method', 'raw_concat')
        self.merged = False

        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

        self.lora_A = nn.ParameterDict()
        for task_id in self.task_ids:
            self.lora_A[str(task_id)] = nn.Parameter(
                torch.zeros(r, in_features), requires_grad=True
            )

        self.lora_B = nn.ParameterDict()
        for task_id in self.task_ids:
            self.lora_B[str(task_id)] = nn.Parameter(
                torch.zeros(out_features, r), requires_grad=True
            )
        
        if self.merge_method == 'learned':
            self.reweight_factor = nn.ParameterDict()
            for task_id in self.task_ids:
                self.reweight_factor[str(task_id)] = nn.Parameter(
                    torch.ones(1) * 0.5, requires_grad=True
                )
        else:
            self.reweight_factor = {} # Changed to a standard dictionary
            for task_id in self.task_ids:
                self.reweight_factor[str(task_id)] = torch.ones(1) * 0.5     
        
        self.reset_parameters()
    
    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        for tid in self.task_ids:
            B = self.lora_B[str(tid)]
            self.reweight_factor[str(tid)].data = self.reweight_factor[str(tid)].data.to(B.device)     

    def __freeze_params__(self, cur_task_id):
        if self.merge_method == 'raw_concat':  
            for task_id in self.task_ids:
                if task_id != cur_task_id:
                    self.lora_A[str(task_id)].requires_grad = False
                    self.lora_B[str(task_id)].requires_grad = False
                    
        elif self.merge_method == 'randA_concat':  
            for task_id in self.task_ids:
                self.lora_A[str(task_id)].requires_grad = False

                if task_id != cur_task_id:
                    self.lora_B[str(task_id)].requires_grad = False
        
        elif self.merge_method == 'learned':  
            pass

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            for task_id in self.task_ids:
                # initialize A with kaiming uniform
                nn.init.kaiming_uniform_(self.lora_A[str(task_id)], a=math.sqrt(5))                          
                # initialize all B matrices to zero
                nn.init.zeros_(self.lora_B[str(task_id)])
    
    def merge_forward(self, x, cur_task_id=None, expert_idx=None):
        input_dtype = x.dtype
        result = 0
        for tid in self.task_ids:
            A = self.lora_A[str(tid)]
            B = self.lora_B[str(tid)]
            weight = self.reweight_factor[str(tid)]
            task_output = (self.lora_dropout(x) @ A.transpose(0, 1) @ B.transpose(0, 1)) * self.scaling
            result += task_output * weight      

        return result.to(input_dtype)
    
    def raw_forward(self, x, cur_task_id=None, expert_idx=None): 
        input_dtype = x.dtype
        
        # # raw forward for training
        if self.training:
            # If task_id not specified and in training, default to latest task
            if cur_task_id is None:
                cur_task_id = self.task_ids[-1]

            A = self.lora_A[str(cur_task_id)]
            B = self.lora_B[str(cur_task_id)]
            result = (self.lora_dropout(x) @ A.transpose(0, 1) @ B.transpose(0, 1)) * self.scaling

        else:  # For inference, average all task outputs
            result = 0
            for tid in self.task_ids:
                A = self.lora_A[str(tid)]
                B = self.lora_B[str(tid)]
                weight = self.reweight_factor[str(tid)]
                task_output = (self.lora_dropout(x) @ A.transpose(0, 1) @ B.transpose(0, 1)) * self.scaling
                result += task_output * weight
        
        return result.to(input_dtype)
    
    def forward(self, x, cur_task_id=None, expert_idx=None): 
        if self.merge_method == 'learned':
            return self.merge_forward(x, cur_task_id, expert_idx)
        else:   
            return self.raw_forward(x, cur_task_id, expert_idx)
        
class LoraLinearRep(nn.Module):
    """LoRA implemented in reparameterized A and B matrix across tasks in the same group"""
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        task_ids: List[int] = None,
        cur_task_id: int = None,
        group_cfg: dict = None,
        is_test: bool = False,
        **kwargs,
    ):  
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        self.task_ids = task_ids if task_ids is not None else []
        self.merge_method = group_cfg.get('merge_method', 'none')
        self.merged = False
        self.lambda_A = group_cfg.get('lambda_A', 0.5)
        self.lambda_B = group_cfg.get('lambda_B', 0.5)
        
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
            
        base_task_id = self.task_ids[0]
        self.lora_A[str(base_task_id)] = nn.Parameter(torch.zeros(r, in_features), requires_grad=True)
        self.lora_B[str(base_task_id)] = nn.Parameter(torch.zeros(out_features, r), requires_grad=True)
        
        if len(task_ids) > 1 and (cur_task_id == task_ids[-1]) and (self.merge_method == 'ema') and (is_test is False) :
            self.lora_A[str(cur_task_id)] = nn.Parameter(torch.zeros(r, in_features), requires_grad=True)
            self.lora_B[str(cur_task_id)] = nn.Parameter(torch.zeros(out_features, r), requires_grad=True)        
        
        self.reset_parameters()
    
    # def __rep__(self, delete):
    #     if len(self.lora_B.keys()) > 1:
    #         base_task_id = self.task_ids[0]
    #         cur_task_id = self.task_ids[-1]
    #         self.lora_A[str(base_task_id)].data = self.lora_A[str(base_task_id)].data * self.lambda_A + \
    #                                                     self.lora_A[str(cur_task_id)].data * (1 - self.lambda_A)
    #         self.lora_B[str(base_task_id)].data = self.lora_B[str(base_task_id)].data * self.lambda_B + \
    #                                                     self.lora_B[str(cur_task_id)].data * (1 - self.lambda_B)
    #         if delete:
    #             del self.lora_A[str(cur_task_id)]
    #             del self.lora_B[str(cur_task_id)]   
    
    def __rep__(self, delete):
        if len(self.lora_B.keys()) > 1:
            base_task_id = self.task_ids[0]
            cur_task_id = self.task_ids[-1]
            self.lora_A[str(cur_task_id)].data = self.lora_A[str(base_task_id)].data * self.lambda_A + \
                                                        self.lora_A[str(cur_task_id)].data * (1 - self.lambda_A)
            self.lora_B[str(cur_task_id)].data = self.lora_B[str(base_task_id)].data * self.lambda_B + \
                                                        self.lora_B[str(cur_task_id)].data * (1 - self.lambda_B)
            if delete:
                self.lora_A[str(base_task_id)].data = self.lora_A[str(cur_task_id)].data
                self.lora_B[str(base_task_id)].data = self.lora_B[str(cur_task_id)].data
                del self.lora_A[str(cur_task_id)]
                del self.lora_B[str(cur_task_id)]   
    
    def __freeze_params__(self, cur_task_id):    
        if len(self.lora_B.keys()) > 1:  # freeze base expert 
            base_task_id = self.task_ids[0]
            self.lora_A[str(base_task_id)].requires_grad = False
            self.lora_B[str(base_task_id)].requires_grad = False            
    
    def reset_parameters(self): 
        for k, v in self.lora_A.items():
            nn.init.kaiming_uniform_(self.lora_A[k], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[k])
    
    # forward base expert in current group, for distillation
    def forward_base_expert(self, x, expert_idx=None):
        input_dtype = x.dtype
        task_id = self.task_ids[0]
        A = self.lora_A[str(task_id)]
        B = self.lora_B[str(task_id)]
        result = (self.lora_dropout(x) @ A.transpose(0, 1) @ B.transpose(0, 1)) * self.scaling                     
        return result.to(input_dtype)          
    
    def forward(self, x, task_id=None, expert_idx=None): 
        input_dtype = x.dtype
        # if self.training:
        #     train_task_id = self.task_ids[-1] if len(self.lora_B.keys()) > 1 else self.task_ids[0]
        #     A = self.lora_A[str(train_task_id)]
        #     B = self.lora_B[str(train_task_id)]
        #     result = (self.lora_dropout(x) @ A.transpose(0, 1) @ B.transpose(0, 1)) * self.scaling
        # else:
        #     val_task_id = self.task_ids[-1] if len(self.lora_B.keys()) > 1 else self.task_ids[0]
        #     A = self.lora_A[str(val_task_id)]
        #     B = self.lora_B[str(val_task_id)]
        #     result = (self.lora_dropout(x) @ A.transpose(0, 1) @ B.transpose(0, 1)) * self.scaling 
        val_task_id = self.task_ids[-1] if len(self.lora_B.keys()) > 1 else self.task_ids[0]
        A = self.lora_A[str(val_task_id)]
        B = self.lora_B[str(val_task_id)]
        result = (self.lora_dropout(x) @ A.transpose(0, 1) @ B.transpose(0, 1)) * self.scaling                     
        return result.to(input_dtype)  
