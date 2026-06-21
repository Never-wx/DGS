import torch
import torch.nn as nn
import torch.nn.functional as F

class MoERouter(nn.Module):
    def __init__(self, 
                 experts_num: int,
                 top_k: int,
                 scoring_func: str = 'softmax',
                 aux_loss_alpha: float = 0.001,
                 seq_aux: bool = True,
                 norm_topk_prob: bool = True,
                 input_transform: str = 'none'):
        super().__init__()
        self.experts_num = experts_num
        self.top_k = top_k
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.input_transform = input_transform
    
    def _transform_input(self, hidden_states, spatial_shapes):
        """  
        1. none mode: Token level routing for both img and text 
        2. mean mode:  Sequence level routing for both img and text
        3. seperate mode: Token level routing for text; Sequence level routing for img
        """
        bsz, seq_len, h = hidden_states.shape
        mean_routing = False   # tokenwise routing w/o mean
        
        if seq_len > 2000:  # visual token
            if self.input_transform in ['mean', 'seperate']:
                last_layer_feat_nums = spatial_shapes[-1][0] * spatial_shapes[-1][1]
                last_layer_start_idx = seq_len - last_layer_feat_nums
                hidden_states = hidden_states[:, last_layer_start_idx:, :].mean(dim=1, keepdim=True)
                mean_routing = True
        else:  # text token
            if self.input_transform == 'mean':
                hidden_states = hidden_states.mean(dim=1, keepdim=True)
                mean_routing = True
        
        return hidden_states.view(-1, h), mean_routing, bsz, seq_len
    
    def _compute_scores(self, hidden_states, w_gate):
        logits = F.linear(hidden_states, w_gate, None)
        if self.scoring_func == 'softmax':
            return logits.softmax(dim=-1)
        raise NotImplementedError(f'不支持的评分函数: {self.scoring_func}')
    
    def _select_topk(self, scores):
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        if self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_weight, topk_idx
    
    def _compute_aux_loss(self, scores, topk_idx, mean_routing, bsz, seq_len):
        if not self.training or self.aux_loss_alpha <= 0.0:
            topk_idx_for_load = topk_idx.view(bsz, -1)
            mask_ce = F.one_hot(topk_idx_for_load.view(-1), num_classes=self.experts_num)
            # load = mask_ce.float().mean(0) * (num_classes * bsz)      
            load = mask_ce.sum(0) 
            return None, load
        
        scores_for_aux = scores
        aux_topk = self.top_k
        topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
        
        if self.seq_aux and not mean_routing:
            # Whether to compute the auxiliary loss for each individual sample.
            scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
            ce = torch.zeros(bsz, self.experts_num, device=scores.device)
            ce.scatter_add_(1, topk_idx_for_aux_loss, 
                          torch.ones(bsz, seq_len * aux_topk, device=scores.device))
            ce.div_(seq_len * aux_topk / self.experts_num)
            aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.aux_loss_alpha
            load = ce.sum(dim=0)
        else:
            mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.experts_num)
            ce = mask_ce.float().mean(0)
            Pi = scores_for_aux.mean(0)
            fi = ce * self.experts_num
            aux_loss = (Pi * fi).sum() * self.aux_loss_alpha
            # load = ce * (self.experts_num * bsz)
            load = mask_ce.sum(0)

        return aux_loss, load
    
    def forward(self, hidden_states, w_gate, spatial_shapes=None):
        # 1. Transform input
        transformed, mean_routing, bsz, seq_len = self._transform_input(hidden_states, spatial_shapes)
        
        # 2. Compute scores
        scores = self._compute_scores(transformed, w_gate)
        
        # 3. Select top-k
        topk_weight, topk_idx = self._select_topk(scores)
        
        # 4. Compute aux loss and load
        aux_loss, load = self._compute_aux_loss(scores, topk_idx, mean_routing, bsz, seq_len)
        
        # 5. Handle mean routing expansion
        if mean_routing:
            topk_idx = topk_idx.unsqueeze(1).expand(-1, seq_len, -1).contiguous().view(bsz*seq_len, -1)
            topk_weight = topk_weight.unsqueeze(1).expand(-1, seq_len, -1).contiguous().view(bsz*seq_len, -1)
        
        return topk_idx, topk_weight, aux_loss, load.detach()

class NoisyMoERouter(MoERouter):
    
    def _compute_scores(self, hidden_states, w_gate, w_noise, train):
        clean_logits = F.linear(hidden_states, w_gate, None)
        if w_noise is not None and train:
            raw_noise_stddev = F.linear(hidden_states, w_noise, None)
            noise_stddev = ((F.softplus(raw_noise_stddev) + 1e-2))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits   
                
        if self.scoring_func == 'softmax':
            return logits.softmax(dim=-1)
        
        raise NotImplementedError(f'不支持的评分函数: {self.scoring_func}')    
    
    def _select_noisy_topk(self, scores):
        top_weight, top_indices = torch.topk(scores, k=min(self.top_k + 1, self.experts_num), dim=-1, sorted=False)
        topk_weight = top_weight[:, :self.top_k].contiguous()
        topk_idx = top_indices[:, :self.top_k].contiguous()
        if self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        
        return topk_weight, topk_idx
            
    def forward(self, hidden_states, w_gate, w_noise=None, train=False, spatial_shapes=None):
        # 1. Transform input
        transformed, mean_routing, bsz, seq_len = self._transform_input(hidden_states, spatial_shapes)
        
        # 2. Compute scores
        scores = self._compute_scores(transformed, w_gate, w_noise, train)
        
        # 3. Select top-k
        if w_noise is not None and train:
            topk_weight, topk_idx = self._select_noisy_topk(scores)
        else:
            topk_weight, topk_idx = self._select_topk(scores)
        
        # 4. Compute aux loss and load
        aux_loss, load = self._compute_aux_loss(scores, topk_idx, mean_routing, bsz, seq_len)
        
        # 5. Handle mean routing expansion
        if mean_routing:
            topk_idx = topk_idx.unsqueeze(1).expand(-1, seq_len, -1).contiguous().view(bsz*seq_len, -1)
            topk_weight = topk_weight.unsqueeze(1).expand(-1, seq_len, -1).contiguous().view(bsz*seq_len, -1)
        
        return topk_idx, topk_weight, aux_loss, load.detach()
    
class CosineMoERouter(MoERouter):
    def _compute_scores(self, hidden_states, w_gate):
        logits = w_gate(hidden_states)
        if self.scoring_func == 'softmax':
            return logits.softmax(dim=-1)
        raise NotImplementedError(f'不支持的评分函数: {self.scoring_func}')

class GlobalRouter(MoERouter):
    def forward(self, hidden_states, task_id, spatial_shapes=None):
        topk_idx = torch.zeros(hidden_states.size(0)*hidden_states.size(1), 1, device=hidden_states.device, dtype=torch.int64)
        topk_weight = torch.ones(hidden_states.size(0)*hidden_states.size(1), 1, device=hidden_states.device, dtype=torch.int64)
        return topk_idx, topk_weight, None, None

class GroupNoisyMoERouter(NoisyMoERouter):
    
    def _select_noisy_topk(self, scores):
        group_experts_num = scores.shape[-1]
        top_weight, top_indices = torch.topk(scores, k=min(self.top_k + 1, group_experts_num), dim=-1, sorted=False)
        topk_weight = top_weight[:, :self.top_k].contiguous()
        topk_idx = top_indices[:, :self.top_k].contiguous()
        if self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        
        return topk_weight, topk_idx
    
    def _compute_aux_loss(self, scores, topk_idx, mean_routing, bsz, seq_len):
        group_experts_num = scores.shape[-1]

        if not self.training or self.aux_loss_alpha <= 0.0:
            topk_idx_for_load = topk_idx.view(bsz, -1)
            mask_ce = F.one_hot(topk_idx_for_load.view(-1), num_classes=group_experts_num)
            # load = mask_ce.float().mean(0) * (self.experts_num * bsz)      
            load = mask_ce.sum(0) 
            return None, load
        
        scores_for_aux = scores
        aux_topk = self.top_k
        topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
        
        if self.seq_aux and not mean_routing:
            # Whether to compute the auxiliary loss for each individual sample.
            scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
            ce = torch.zeros(bsz, group_experts_num, device=scores.device)
            ce.scatter_add_(1, topk_idx_for_aux_loss, 
                          torch.ones(bsz, seq_len * aux_topk, device=scores.device))
            ce.div_(seq_len * aux_topk / group_experts_num)
            aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.aux_loss_alpha
            load = ce.sum(dim=0)
        else:
            mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=group_experts_num)
            ce = mask_ce.float().mean(0)
            Pi = scores_for_aux.mean(0)
            fi = ce * group_experts_num
            aux_loss = (Pi * fi).sum() * self.aux_loss_alpha
            # load = ce * (self.experts_num * bsz)
            load = mask_ce.sum(0)

        return aux_loss, load 

class BatchPrioritizedCosineRouter(CosineMoERouter):
    def __init__(self, 
                 keep_ratio: int = 0.9,
                 sort_func: str = 'max',
                 **kwargs):
        super().__init__(**kwargs)
        self.keep_ratio = keep_ratio
        self.sort_func = sort_func

    def _select_topk(self, scores):
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        if self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        
        # Drop tokens with low weight
        if self.sort_func == 'max':
            pass
        elif self.sort_func == 'sum':
            pass

        return topk_weight, topk_idx
    
    def forward(self, hidden_states, w_gate, spatial_shapes=None):
        # 1. Transform input
        transformed, mean_routing, bsz, seq_len = self._transform_input(hidden_states, spatial_shapes)
        
        # 2. Compute scores
        scores = self._compute_scores(transformed, w_gate)
        
        # 3. Select top-k
        topk_weight, topk_idx = self._select_topk(scores)
        
        # 4. Compute aux loss and load
        aux_loss, load = self._compute_aux_loss(scores, topk_idx, mean_routing, bsz, seq_len)
        
        # 5. Handle mean routing expansion
        if mean_routing:
            topk_idx = topk_idx.unsqueeze(1).expand(-1, seq_len, -1).contiguous().view(bsz*seq_len, -1)
            topk_weight = topk_weight.unsqueeze(1).expand(-1, seq_len, -1).contiguous().view(bsz*seq_len, -1)

        return topk_idx, topk_weight, aux_loss, load.detach()