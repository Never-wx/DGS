import copy
import torch
import torch.nn.functional as F
from torch import Tensor, nn
zero_value = 1e-8
lan_scale = 0.1
vis_scale = 0.1

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
    
class RepLinearLayer(nn.Linear):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = False,
        aux_loss='SmoothL1Loss',
        **kwargs
    ):  
        super().__init__(in_features, out_features, bias)    # rdb_fast_branch
        self.scaling = nn.parameter.Parameter(torch.ones(1) * lan_scale)
        nn.init.constant_(self.weight, val=zero_value)
        self.freeze_linear = nn.Linear(in_features, out_features, bias)  # rdb_slow_branch
        nn.init.constant_(self.freeze_linear.weight, val=0.0)
        if self.bias is not None:
            nn.init.constant_(self.freeze_linear.bias, val=0.0) 
        
        if aux_loss is not None:
            if aux_loss == 'SmoothL1Loss':
                self.zero_inter_loss = torch.nn.SmoothL1Loss(reduction='mean')
            elif aux_loss == 'L1Loss':
                self.zero_inter_loss = torch.nn.L1Loss(reduction='mean')
            elif aux_loss == 'MSELoss':
                self.zero_inter_loss = torch.nn.MSELoss(reduction='mean')
            else:
                raise ValueError(f"Unsupported auxiliary loss: {aux_loss}")
        else:
            self.zero_inter_loss = None

    def forward(self, input: Tensor) -> Tensor:
        if self.training :
            branch_output = self.scaling * super().forward(input)
            output = branch_output + self.freeze_linear(input)
            if self.zero_inter_loss is not None:
                aux_loss = self.zero_inter_loss(branch_output, torch.zeros_like(branch_output)) + \
                    self.zero_inter_loss(output, torch.zeros_like(output))
                aux_loss = aux_loss * 0.1
                output = AddAuxiliaryLoss.apply(output, aux_loss)
            return output
        else:
            return self.freeze_linear(input)
        
    def __rep__(self, delete=False):
        self.freeze_linear.weight.data = self.weight.data  * self.scaling + self.freeze_linear.weight.data
        self.scaling = nn.parameter.Parameter(torch.ones(1).to(self.weight.data) * lan_scale)
        nn.init.constant_(self.weight, val=zero_value)
        
        if self.bias is not None:
            self.freeze_linear.bias.data = self.bias.data * self.scaling + self.freeze_linear.bias.data
            nn.init.constant_(self.bias, val=zero_value)
     
class RepLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, **kwargs):
        nn.Linear.__init__(self, in_features, out_features)   # main_branch
        self.rdb_layer = RepLinearLayer(in_features=in_features, out_features=out_features, bias=bias, **kwargs)
        self.reset_parameters()
        # freeze main_branch
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        self.merged = False
        self.merge_weights = False

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                self.weight.data -= self.rdb_layer.freeze_linear.weight.data
                self.bias.data -= self.rdb_layer.freeze_linear.bias.data
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                self.weight.data += self.rdb_layer.freeze_linear.weight.data    # BUG: before last epoch, ignored the output of fast layer
                self.bias.data += self.rdb_layer.freeze_linear.bias.data
                self.merged = True       
    
    def forward(self, input: torch.Tensor):
        if not self.merged:
            main_output = F.linear(input, self.weight, bias=self.bias)    
            rdb_output  = self.rdb_layer(input)       
            output = main_output + rdb_output
            return output
        else:
            return F.linear(input, self.weight, bias=self.bias)        