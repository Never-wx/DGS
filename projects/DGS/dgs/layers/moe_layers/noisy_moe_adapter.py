# The code is based on  https://github.com/JiazuoYu/MoE-Adapters4CL
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from ..modules.adapter import Adapter
import math
from .base_moe import AddAuxiliaryLoss

class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # print(self._num_experts)
        # sort experts
        # print('gates', gates.shape) # 64, 22
        # [[0.0000, 0.0000, 0.5146, 0.4854, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        #         [0.0000, 0.0000, 0.0000, 0.0000, 0.4666, 0.5334, 0.0000, 0.0000, 0.0000]]
        # print(torch.nonzero(gates).shape)  # torch.Size([128, 2])
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)

        # print(sorted_experts.shape, index_sorted_experts.shape) # torch.Size([128, 2]) torch.Size([128, 2])
        # [[0, 2],[0, 3],[1, 4],[1, 5]] sorted_experts 将feature和experts匹配上
        # [[1, 0],[0, 1],[2, 2],[3, 3]]

        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # print(self._batch_index)
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero
        # expand according to batch index so we can just split by _part_sizes

        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space

        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)  # weight


        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), device=stitched.device)
        # combine samples that have been processed by the same k experts

        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # back to log space
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class NoisyMoEAdapter(nn.Module):
    def __init__(self, in_features, out_features, bottleneck=64, dropout=0.1, scale=1.0, 
                 init_option='lora', adapter_layernorm_option='none', experts_num=20, top_k=2, 
                 noisy_gating=True, input_transform='none', num_tasks=10, task_id=-1, **kwargs):
        
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features   
        self.num_tasks = num_tasks
        self.task_id = task_id
        self.val_task_id = None
        self.spatial_shapes = None

        self.experts_num = experts_num
        self.top_k = top_k
        self.noisy_gating = noisy_gating
        self.adapter_layernorm_option = adapter_layernorm_option
        self.input_transform = input_transform
        # 初始化专家路由参数
        self.gates = nn.ParameterList()
        self.w_noise_list = nn.ParameterList()
        for _ in range(self.num_tasks):  # 任务数量
            self.gates.append(nn.Parameter(torch.zeros(in_features, experts_num), requires_grad=True))
            self.w_noise_list.append(nn.Parameter(torch.zeros(in_features, experts_num), requires_grad=True))
        
        # 初始化专家适配器
        self.experts = nn.ModuleList()
        for _ in range(experts_num):
            adapter = Adapter(
                d_model=in_features,
                dropout=dropout,
                bottleneck=bottleneck,
                init_option=init_option,
                adapter_scalar=scale,
                adapter_layernorm_option=adapter_layernorm_option
            )
            self.experts.append(adapter)
            
        # 初始化统计参数
        self.register_buffer("mean", torch.tensor([0.0]), persistent=False)
        self.register_buffer("std", torch.tensor([1.0]), persistent=False)
        self.register_buffer("activate_rate", torch.zeros([self.experts_num]), persistent=False)
        self.register_buffer("frozen_flags", torch.zeros([self.experts_num]), persistent=False)
        self.activate_count_flag = False
        self.softmax = nn.Softmax(dim=1)
        self.softplus = nn.Softplus()
        
        # 初始化权重
        with torch.no_grad():
            for param in self.gates:
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            for param in self.w_noise_list:
                nn.init.zeros_(param)

        self.__freeze_params__()
    
    def __set_activate_count_flags__(self):
        self.activate_count_flag = True

    def __set_frozen_flags__(self):
        k_act_nums, k_act_id  = torch.topk(self.activate_rate, k=self.top_k)
        self.frozen_flags[k_act_id] = 1

    def __freeze_params__(self):
        for i in range(self.num_tasks):
            if i != self.task_id:
                self.gates[i].requires_grad_(False)
                self.w_noise_list[i].requires_grad_(False)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """        
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.top_k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        return torch.where(is_in, prob_if_in, prob_if_out)

    def noisy_top_k_gating(self, x, train, w_gate, w_noise, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """        
        clean_logits = x @ w_gate.to(x)
        if self.noisy_gating and train:
            raw_noise_stddev = x @ w_noise.to(x)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.top_k + 1, self.experts_num), dim=1)
        top_k_logits = top_logits[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        
        if self.noisy_gating and self.top_k < self.experts_num and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def set_eval_task_id(self, task_id: int):
        """Set task ID for evaluation"""
        self.val_task_id = task_id
        
    def set_spatial_shapes(self, spatial_shapes: torch.Tensor):
        """Set spatial shapes for routing"""
        self.spatial_shapes = spatial_shapes

    def forward(self, x, add_residual=False):
        B, L, D = x.shape
          
        if L > 2000:
            if self.input_transform == 'mean' or self.input_transform == 'seperate':
                last_layer_feat_nums = self.spatial_shapes[-1][0] * self.spatial_shapes[-1][1] 
                last_layer_start_idx = L - last_layer_feat_nums
                gates_input = x[:, last_layer_start_idx:, :].mean(dim=1)
                x_dispatch = x
            else:
                gates_input = x.view(-1, D)
                x_dispatch = gates_input
        else:   # text_token
            if self.input_transform == 'mean':
                gates_input = x.mean(dim=1)
                x_dispatch = x
            else:
                gates_input = x.view(-1, D)
                x_dispatch = gates_input     
                
        if self.training:   
            assert self.task_id > -1, "self.task_id must be set during training"
            task_id = self.task_id
        else:
            assert self.val_task_id is not None, "val_task_id must be set in evaluation mode"  
            task_id = self.val_task_id
            if task_id == -1:   # zero-shot
                return torch.zeros_like(x)  
        
        gates, load = self.noisy_top_k_gating(
            gates_input, 
            self.training, 
            self.gates[task_id] if self.num_tasks > 1 else self.gates[0],   
            self.w_noise_list[task_id] if self.num_tasks > 1 else self.w_noise_list[0],
        )   # [B*L, topk]
        
        if self.activate_count_flag:
            self.activate_rate = self.activate_rate + load.detach()
        
        dispatcher = SparseDispatcher(self.experts_num, gates)
        expert_inputs = dispatcher.dispatch(x_dispatch)
        
        expert_outputs = []
        # for i in range(self.experts_num):
        #     if expert_inputs[i].shape[0] > 0:  # avoid updating experts which are not selected
        #         out = self.experts[i](expert_inputs[i], add_residual=add_residual)
        #         expert_outputs.append(out.view(out.size(0), -1))
        
        for i in range(self.experts_num):
            out = self.experts[i](expert_inputs[i], 
                                  add_residual=add_residual)
            if out.shape[0] > 0:  
                expert_outputs.append(out.view(out.shape[0], -1))
        
        combined = dispatcher.combine(expert_outputs)
        output = combined.view_as(x)
        
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= 1e-2

        output = AddAuxiliaryLoss.apply(output, loss)

        return output


