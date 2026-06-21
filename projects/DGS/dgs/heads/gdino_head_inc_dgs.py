import copy
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import Linear
from mmengine.model import constant_init
from mmengine.structures import InstanceData
from torch import Tensor
from mmengine import Config
from mmdet.models.losses import QualityFocalLoss
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.utils import InstanceList, ConfigType, reduce_mean,  OptInstanceList
from mmdet.models.utils import multi_apply
from mmdet.structures import SampleList
from .gdino_head_inc_gcd import GroundingDINOHead_inc_gcd

@MODELS.register_module(name='GroundingDINOHead_inc_DGS')
class GroundingDINOHead_inc_DGS(GroundingDINOHead_inc_gcd):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def loss(self, new_head_inputs_dict, 
             old_head_inputs_dict, 
             ori_head_inputs_dict, 
             batch_data_samples) -> dict:       
         
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        # self.ori_text_mask = ori_head_inputs_dict['ori_text_token_mask']  # text_mask for old class 
        self.ori_topk_query = ori_head_inputs_dict['ori_topk_query']
        self.ori_token_positive_maps = ori_head_inputs_dict['ori_token_positive_maps'] 

        self.token_positive_maps = new_head_inputs_dict['token_positive_maps']
        self.text_masks = new_head_inputs_dict['text_token_mask']   # text mask for new class  
        self.ori_text_masks = old_head_inputs_dict['text_token_mask']   # text_mask for old class 

        new_outs = self(new_head_inputs_dict['hidden_states'], 
                        new_head_inputs_dict['references'], 
                        new_head_inputs_dict['memory_text'], 
                        new_head_inputs_dict['text_token_mask'])  
        
        old_outs = self(old_head_inputs_dict['hidden_states'], 
                        old_head_inputs_dict['references'], 
                        old_head_inputs_dict['memory_text'], 
                        old_head_inputs_dict['text_token_mask'])  
        
        new_enc_cls_scores = new_head_inputs_dict['enc_outputs_class']
        new_enc_bbox_preds = new_head_inputs_dict['enc_outputs_coord']
        dn_meta = new_head_inputs_dict['dn_meta']

        # old_enc_cls_scores = old_head_inputs_dict['enc_outputs_class']
        # old_enc_bbox_preds = new_head_inputs_dict['enc_outputs_coord']
        hidden_states = old_head_inputs_dict['hidden_states']
        memory_text = old_head_inputs_dict['memory_text']

        all_layers_ori_cls_scores = ori_head_inputs_dict['all_layers_ori_cls_scores']
        all_layers_ori_bbox_preds = ori_head_inputs_dict['all_layers_ori_bbox_preds']
        ori_hidden_states = ori_head_inputs_dict['ori_hidden_states']
        ori_memory_text = ori_head_inputs_dict['ori_memory_text']
        # ori_enc_outputs_class = ori_head_inputs_dict['enc_outputs_class']
        # ori_enc_outputs_coord = ori_head_inputs_dict['enc_outputs_coord']
        if 'batch_pseudo_instances' in ori_head_inputs_dict.keys():
            batch_pseudo_instances = ori_head_inputs_dict['batch_pseudo_instances']
            batch_all_instances = ori_head_inputs_dict['batch_all_instances']
        else:
            batch_pseudo_instances = None
            batch_all_instances = None

        detr_loss_inputs = new_outs + (new_enc_cls_scores, new_enc_bbox_preds, 
                                       dn_meta, batch_gt_instances, batch_img_metas, batch_all_instances)
        
        distn_loss_inputs = old_outs + (batch_gt_instances, batch_img_metas, hidden_states, memory_text) + \
                                        (all_layers_ori_cls_scores, all_layers_ori_bbox_preds,
                                         batch_pseudo_instances, batch_all_instances, ori_hidden_states,
                                         ori_memory_text)
        
        loss_dict_old = self.loss_by_feat_old(*distn_loss_inputs)
        loss_dict_new = self.loss_by_feat_new(*detr_loss_inputs)
        loss_dict_new.update(loss_dict_old)
        return loss_dict_new        
    
