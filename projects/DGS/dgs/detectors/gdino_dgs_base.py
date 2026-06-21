# Copyright (c) OpenMMLab. All rights reserved.
import copy
import re
import warnings
from typing import Dict, Optional, Tuple, Union
import os
import torch
import torch.nn as nn
from torch import Tensor
from mmengine.logging import MMLogger
from collections import OrderedDict
from mmdet.registry import MODELS
from mmengine import Config
from mmengine.model import is_model_wrapper
from mmengine.runner import load_checkpoint, load_state_dict
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType
from mmdet.models.layers import SinePositionalEncoding, CdnQueryGenerator
from mmdet.models.layers.transformer.grounding_dino_layers import GroundingDinoTransformerEncoder, GroundingDinoTransformerDecoder
from .gdino_inc import GroundingDINO_inc
from ..layers import GroundingDinoTransformerEncoder_MoE, GroundingDinoTransformerDecoder_MoE
from ..domain_predictor import AdaptiveDomainPredictor

@MODELS.register_module()
class GroundingDINO_DGS_Base(GroundingDINO_inc):
    """Implementation of `Grounding DINO: Marrying DINO with Grounded Pre-
    Training for Open-Set Object Detection.
    
    <https://arxiv.org/abs/2303.05499>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/GroundingDINO>`_.
    """
    
    def __init__(self, 
                 num_tasks=None, 
                 task_id=None,
                 seen_tasks=None,
                 domain_predictor_cfg: OptConfigType=None, 
                 moe_cfg: OptConfigType=None, 
                 *args, **kwargs) -> None:
        
        self.domain_predictor_cfg = domain_predictor_cfg
        self.num_tasks = num_tasks
        self.task_id = task_id
        self.moe_cfg = moe_cfg if moe_cfg is not None else Config._dict_to_config_dict_lazy(dict(type='none'))
        self.moe_cfg["num_tasks"]= num_tasks
        self.moe_cfg["task_id"]= task_id 
        if seen_tasks is not None:
            if type(seen_tasks) is not tuple and type(seen_tasks) is str:
                    seen_tasks = tuple(seen_tasks.split(','))
        self.seen_tasks = seen_tasks 
        super().__init__(*args, **kwargs)
        
    def _freeze_stages_init(self):
        exclude_keywords = self.frozen_cfg.exclude_keywords if 'exclude_keywords' in self.frozen_cfg \
            else ['lora_', 'adapter', 'experts', 'gates', 'rdb']
        
        if self.frozen_cfg.all_frozen:
            for key, p in self.named_parameters():
                if not any(exclude_keyword in key for exclude_keyword in exclude_keywords):
                    p.requires_grad = False
        
        if self.frozen_cfg.backbone_frozen:
            for key, p in self.backbone.named_parameters():
                p.requires_grad = False
            self.level_embed.requires_grad = False

        if self.frozen_cfg.language_model_frozen:
            for key, p in self.language_model.named_parameters():
                p.requires_grad = False
            for key, p in self.text_feat_map.named_parameters():
                p.requires_grad = False
            
        if self.frozen_cfg.neck_frozen:
            for key, p in self.neck.named_parameters():
                p.requires_grad = False

        if self.frozen_cfg.encoder_frozen:
            for key, p in self.encoder.named_parameters():
                if not any(exclude_keyword in key for exclude_keyword in exclude_keywords):
                    p.requires_grad = False
        
        if self.frozen_cfg.decoder_frozen:
            for key, p in self.decoder.named_parameters():
                if not any(exclude_keyword in key for exclude_keyword in exclude_keywords):
                    p.requires_grad = False
            
            for key, p in self.query_embedding.named_parameters():
                p.requires_grad = False

            for key, p in self.memory_trans_fc.named_parameters():
                p.requires_grad = False   

            for key, p in self.memory_trans_norm.named_parameters():
                p.requires_grad = False 

        if self.frozen_cfg.head_frozen:
            for key, p in self.bbox_head.named_parameters():
                p.requires_grad = False 
            
            # for key, p in self.dn_query_generator.named_parameters():
            #     p.requires_grad = False  
        
        if 'freeze_topk_activate' in self.frozen_cfg:        
            logger: MMLogger = MMLogger.get_current_instance()
            work_dir = os.path.dirname(os.path.dirname(os.path.dirname(logger.log_file)))
            frozen_path = os.path.join(work_dir, 'frozen_experts.txt')
            if os.path.isfile(frozen_path):
                with open(frozen_path, "r") as file:
                    lines = file.read().splitlines()
                    frozen_list = list(set(lines))
                    for k, v in self.named_parameters():
                        if k in frozen_list:
                            v.requires_grad = False  
            else:
                print(f"Frozen experts file not found at {frozen_path}")
    
    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.language_model = MODELS.build(self.language_model_cfg)
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = GroundingDinoTransformerEncoder_MoE(**self.encoder, moe_cfg=self.moe_cfg)
        self.decoder = GroundingDinoTransformerDecoder_MoE(**self.decoder, moe_cfg=self.moe_cfg)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'
        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.text_feat_map = nn.Linear(
            in_features=self.language_model.language_backbone.body.language_dim,
            out_features=self.embed_dims,
            bias=True)
            
        self.moe_modules = []
        for _, module in self.named_modules():
            if hasattr(module, 'val_task_id'):  
                self.moe_modules.append(module)

        # global router
        if self.domain_predictor_cfg is not None:   
            self.domain_predictor = AdaptiveDomainPredictor(
                self.domain_predictor_cfg,                                                              
                num_tasks=self.num_tasks, 
                moe_modules=self.moe_modules,
                task_id=self.task_id, 
                seen_tasks=self.seen_tasks)
    
    def set_moe_task_id(self, task_id):
        # transform task_id to group_id
        if hasattr(self, 'domain_predictor') and hasattr(self.domain_predictor, 'task_id_mapping') and task_id >= 0:
            task_id = self.domain_predictor.task_id_mapping[task_id] 
        
        for module in self.moe_modules:
            module.set_eval_task_id(task_id)
    
    def set_spatial_shapes(self, spatial_shapes):
        for module in self.moe_modules:
            module.set_spatial_shapes(spatial_shapes)
    
    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(GroundingDINO_inc, self).train(mode)

    def init_weights(self) -> None:
        pass
        
    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor, valid_ratios: Tensor,
                        text_dict: Dict,
                        batch_data_samples=None) -> Dict:
        text_token_mask = text_dict['text_token_mask']
        memory, memory_text = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # for text encoder
            memory_text=text_dict['embedded'],
            text_attention_mask=~text_token_mask,
            position_ids=text_dict['position_ids'],
            text_self_attention_masks=text_dict['masks'])
        
        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask)
        return encoder_outputs_dict
    
    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        # stack the img feat, 4*[B,C,H_lvl,W_lvl] = [B,C,L], where L = (88*123)+(44*62)+(22*31)+(11*16)
        # encoder_inputs_dict = {'feat', 'feat_mask', 'spatial_shapes', 'level_start_index'}
        # feat = [B,C,L], feat_mask=None, spatial_shapes=[(88,123),(44,62),(22,31),(11,16)], lvl_start_index=[0, 88*123, ……]
        # encoder_outputs_dict['memory'].size = [B,L,D], where L=feat.L
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        self.set_spatial_shapes(encoder_inputs_dict['spatial_shapes'])

        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)
        
        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        
        decoder_inputs_dict.update(tmp_dec_in)
        
        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)

        return head_inputs_dict
    
    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: OptSampleList = None,
        query_distn: OptConfigType = None
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        
        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory, memory_text,
                                     text_token_mask)
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals
        
        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]

        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training :
            if self.dn_cfg is not None:
                dn_label_query, dn_bbox_query, dn_mask, dn_meta = self.dn_query_generator(batch_data_samples)
                query = torch.cat([dn_label_query, query], dim=1)
                reference_points = torch.cat([dn_bbox_query, topk_coords_unact], dim=1)
            else:
                dn_mask, dn_meta = None, None                
                reference_points = topk_coords_unact
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        if query_distn is None:
            query_distn = Config._dict_to_config_dict_lazy(dict(type='None'))
        
        if self.training  or query_distn.type != 'None':
            head_inputs_dict = dict(enc_outputs_class=topk_score, enc_outputs_coord=topk_coords, 
                                    dn_meta=dn_meta) 
        else:
            head_inputs_dict = dict()

        # append text_feats to head_inputs_dict
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        return decoder_inputs_dict, head_inputs_dict
    
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        # TODO: Only open vocabulary tasks are supported for training now.

        # load all the 80 cls token as text_prompts
        text_prompts = [
            data_samples.text for data_samples in batch_data_samples
        ]
        
        gt_labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples
        ]
        # BUG：当只给后40类标签的时候，gt_label会被映射到1~40
        if 'tokens_positive' in batch_data_samples[0]:
            tokens_positive = [
                data_samples.tokens_positive
                for data_samples in batch_data_samples
            ]
            positive_maps = []
            for token_positive, text_prompt, gt_label in zip(
                    tokens_positive, text_prompts, gt_labels):
                tokenized = self.language_model.tokenizer(
                    [text_prompt],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                new_tokens_positive = [
                    token_positive[label.item()] for label in gt_label
                ]
                _, positive_map = self.get_positive_map(
                    tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
            new_text_prompts = text_prompts
        else:
            new_text_prompts = []
            positive_maps = []
            # 在仅使用COCO时：
            #   Batch = 2, text_prompts.size = [2]，len(text_prompts[0]) = 80 = len(text_prompts[1])
            #   set(text_prompts)过滤掉重复的cls, len(set(text_prompts)) = 1
            if len(set(text_prompts)) == 1:
                # All the text prompts are the same,
                # so there is no need to calculate them multiple times.
                tokenized, caption_string, tokens_positive, _ = \
                    self.get_tokens_and_prompts(
                        text_prompts[0], True)
                # tokenized{''input_ids':[[101, 2711, ..]], 'token_type_ids':[[0, 0, ..]], 'attention_mask':[[1, 1, ..]]}
                # the input_ids represent the cls id in the tokenizer codebook.
                new_text_prompts = [caption_string] * len(batch_inputs)
                for gt_label in gt_labels:
                    # map the gt_label from id to position in the caption, namley [14] -> [131, 135]
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    # NOTE construct a map such that positive_map[i,j] = True if box i is associated to token j
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
            else:
                for text_prompt, gt_label in zip(text_prompts, gt_labels):
                    tokenized, caption_string, tokens_positive, _ = \
                        self.get_tokens_and_prompts(
                            text_prompt, True)
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
                    new_text_prompts.append(caption_string)
        
        # text_dict = ['embedded':[B, 195, 256], 'masks':[B, 195, 195], 'hidden':[B,195,768],
        # 'position_ids':[B, 195], 'text_token_masks':[B, 195]]
        # NOTE 195 is the length of the caption including 'cls' + '.' + 'SOS' + 'EOS', and some cls like
        # traffic light will be mapped to 2 ids, therefore its length is 2.

        text_dict = self.language_model(new_text_prompts)
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])   # [in_feat = 768, out_feat = 256]
        
        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(
                batch_inputs.device).bool().float()
            text_token_mask = text_dict['text_token_mask'][i]
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = \
                text_token_mask.unsqueeze(0).repeat(
                    len(positive_map), 1)
            
        visual_features = self.extract_feat(batch_inputs)  # 4*[B, C, H_lvl, W_lvl], [h,w]=[(88,123),(44,62),(22,31),(11,16)]

        head_inputs_dict = self.forward_transformer(visual_features, text_dict, batch_data_samples)
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)

        if self.domain_predictor_cfg is not None:
            loss_domain_pred = self.domain_predictor(visual_features, task_id=self.task_id)
            if loss_domain_pred:
                losses.update(loss_domain_pred)

        return losses
    
    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
        for data_samples in batch_data_samples:
            text_prompts.append(data_samples.text)
            if 'caption_prompt' in data_samples:
                enhanced_text_prompts.append(data_samples.caption_prompt)
            else:
                enhanced_text_prompts.append(None)
            tokens_positives.append(data_samples.get('tokens_positive', None))
        
        if 'custom_entities' in batch_data_samples[0]:
            # Assuming that the `custom_entities` flag
            # inside a batch is always the same. For single image inference
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False
        if len(text_prompts) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompts[0], custom_entities, enhanced_text_prompts[0],
                    tokens_positives[0])
            ] * len(batch_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompt,
                                                     custom_entities,
                                                     enhanced_text_prompt,
                                                     tokens_positive)
                for text_prompt, enhanced_text_prompt, tokens_positive in zip(
                    text_prompts, enhanced_text_prompts, tokens_positives)
            ]
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts)
        
        # image feature extraction
        visual_feats = self.extract_feat(batch_inputs) 
        
        # select router
        if self.domain_predictor_cfg is not None:
            task_id = self.domain_predictor(visual_feats, input_sample=batch_data_samples)
        else:
            task_id = self.task_id
        
        self.set_moe_task_id(task_id)
        
        if isinstance(text_prompts[0], list):
            # chunked text prompts, only bs=1 is supported
            assert len(batch_inputs) == 1
            batch_data_samples = self.fast_chunked_predict(text_prompts, token_positive_maps, visual_feats, 
                                                            entities, batch_data_samples) 
        else:
            # extract text feats
            text_dict = self.language_model(list(text_prompts))
            # text feature map layer
            if self.text_feat_map is not None:
                text_dict['embedded'] = self.text_feat_map(
                    text_dict['embedded'])

            is_rec_tasks = []
            
            for i, data_samples in enumerate(batch_data_samples):
                if token_positive_maps[i] is not None:
                    is_rec_tasks.append(False)
                else:
                    is_rec_tasks.append(True)
                data_samples.token_positive_map = token_positive_maps[i]
                
            # head_inputs_dict = [memory_text, text_token_mask, hidden_states, references]
            # memory_text.size = [1, 195, 256], text_token_mask = [1, 195]
            # hidden_states.size = [6, 1 900, 256], references.size = 7 * [1, 900, 4]
            head_inputs_dict = self.forward_transformer(
                visual_feats, text_dict, batch_data_samples)
                       
            results_list = self.bbox_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples)  
                
            for data_sample, pred_instances, entity, is_rec_task in zip(
                    batch_data_samples, results_list, entities, is_rec_tasks):
                if len(pred_instances) > 0:
                    label_names = []
                    for labels in pred_instances.labels:
                        if is_rec_task:
                            label_names.append(entity)
                            continue
                        if labels >= len(entity):
                            warnings.warn(
                                'The unexpected output indicates an issue with '
                                'named entity recognition. You can try '
                                'setting custom_entities=True and running '
                                'again to see if it helps.')
                            label_names.append('unobject')
                        else:
                            label_names.append(entity[labels])
                    # for visualization
                    pred_instances.label_names = label_names
                data_sample.pred_instances = pred_instances
        
        return batch_data_samples

