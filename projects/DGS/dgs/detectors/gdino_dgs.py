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
from ..losses import generate_distn_points
from .gdino_dgs_base import GroundingDINO_DGS_Base

@MODELS.register_module()
class GroundingDINO_DGS(GroundingDINO_DGS_Base):
    def __init__(self, distn_cfg,  vis_cfg: OptConfigType = None, *args, **kwargs) -> None:
        self.distn_cfg = distn_cfg
        if 'feat_distn' not in self.distn_cfg:
            self.distn_cfg.feat_distn = Config._dict_to_config_dict_lazy(dict(type='None'))
        if 'query_distn' not in self.distn_cfg:
            self.distn_cfg.query_distn = Config._dict_to_config_dict_lazy(dict(type='None'))
        
        self.vis_cfg = vis_cfg if vis_cfg is not None else Config._dict_to_config_dict_lazy(dict(type='none'))
        super().__init__(*args, **kwargs)
        # self.start=self.bbox_head.trunc_class[0]
        # self.end=self.bbox_head.trunc_class[1]
        self.max_text_len=self.language_model.max_tokens
        self.token_positive_maps=None
        self.ori_token_positive_maps=None
        self.new_text_token_mask_chunked=None
        # self.load_base_detector()
    
    def set_switch_signal(self, signal):
        for module in self.moe_modules:
            if hasattr(module, 'set_switch_signal'):
                module.set_switch_signal(signal)

    def load_base_detector(self):
        ori_cfg = Config.fromfile(self.distn_cfg['ori_config_file'])
        ori_model = MODELS.build(ori_cfg.model)
        ori_model.eval()
        for param in ori_model.parameters():
            param.requires_grad = False
        self.ori_model = ori_model
    
    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
        aux_dict: Dict = None
    ) -> Dict:
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)
        
        if self.training:
            new_decoder_inputs_dict = decoder_inputs_dict.copy()
            old_decoder_inputs_dict = decoder_inputs_dict.copy()
            # forward on newtext
            new_tmp_dec_in, new_head_inputs_dict = self.pre_decoder(
                **encoder_outputs_dict, batch_data_samples=batch_data_samples)
            new_decoder_inputs_dict.update(new_tmp_dec_in)
            new_decoder_outputs_dict = self.forward_decoder(**new_decoder_inputs_dict)
            new_head_inputs_dict.update(new_decoder_outputs_dict)

            # forward on oldtext
            encoder_outputs_dict['text_token_mask'] = self.ori_text_masks  

            old_tmp_dec_in, old_head_inputs_dict = self.pre_decoder_old(
                **encoder_outputs_dict, aux_dict=aux_dict, batch_data_samples=batch_data_samples)
            old_decoder_inputs_dict.update(old_tmp_dec_in)
            old_decoder_outputs_dict = self.forward_decoder(**old_decoder_inputs_dict)
            old_head_inputs_dict.update(old_decoder_outputs_dict)

            return new_head_inputs_dict, old_head_inputs_dict
        
        else:
            tmp_dec_in, head_inputs_dict = self.pre_decoder(
                **encoder_outputs_dict, batch_data_samples=batch_data_samples)
            decoder_inputs_dict.update(tmp_dec_in)      
            decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
            head_inputs_dict.update(decoder_outputs_dict)

            return head_inputs_dict      
    
    def pre_decoder_old(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        aux_dict: Dict = None,
        batch_data_samples: OptSampleList = None,
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
        topk_indices = torch.topk(enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]
        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()

        aux_query, aux_reference, self_attn_mask = generate_distn_points(self.distn_cfg, aux_dict)   
        query = aux_query
        reference_points = aux_reference

        dn_mask, dn_meta = None, None
        
        # reference_points = reference_points.sigmoid()

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
        if self.training :
            head_inputs_dict = dict(enc_outputs_class=topk_score, enc_outputs_coord=topk_coords, 
                                    dn_meta=dn_meta) 
        else:
            head_inputs_dict = dict()
        # append text_feats to head_inputs_dict
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        return decoder_inputs_dict, head_inputs_dict  
        
    @torch.no_grad()
    def forward_with_expert_switch(
            self,
            img_feats: Tuple[Tensor],
            text_dict: Dict,
            batch_data_samples: OptSampleList = None,
        ):
        self.set_switch_signal(signal=True)
        self.set_moe_task_id(task_id=self.task_id)
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)
        
        self.set_spatial_shapes(encoder_inputs_dict['spatial_shapes'])

        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)
        
        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples, query_distn=self.distn_cfg.query_distn)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)  
        self.set_switch_signal(signal=False)

        if self.distn_cfg.query_distn.type == 'seperate_queryinit' :
            head_inputs_dict['aux_query'] = tmp_dec_in['query'].clone()
            head_inputs_dict['aux_reference'] = tmp_dec_in['reference_points'].clone()

        head_inputs_dict['ori_text_token_mask'] = head_inputs_dict.pop('text_token_mask')
        head_inputs_dict['ori_memory_text'] = head_inputs_dict.pop('memory_text')
        head_inputs_dict['ori_hidden_states'] = head_inputs_dict.pop('hidden_states')
        head_inputs_dict['ori_references'] = head_inputs_dict.pop('references')
        return head_inputs_dict
    
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        
        text_prompts = [
            data_samples.text for data_samples in batch_data_samples
        ]
        # text for ori model 
        ori_text_prompts = [
            data_samples.ori_text for data_samples in batch_data_samples
        ]

        gt_labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples
        ]
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

            if len(set(text_prompts)) == 1:
                tokenized, caption_string, tokens_positive, _ = \
                    self.get_tokens_and_prompts(
                        text_prompts[0], True)

                new_text_prompts = [caption_string] * len(batch_inputs)

                ori_tokenized, ori_caption_string, ori_tokens_positive, _ = \
                    self.get_tokens_and_prompts(
                        ori_text_prompts[0], True)
                ori_text_prompts = [ori_caption_string] * len(batch_inputs)

                # generate global ori_token_positive_maps and ori model prompts
                if self.token_positive_maps is None:
                    token_positive_maps, _ = self.get_positive_map(tokenized, tokens_positive)
                    self.token_positive_maps = token_positive_maps
                    ori_token_positive_maps, _ = self.get_positive_map(ori_tokenized, ori_tokens_positive)
                    self.ori_token_positive_maps = ori_token_positive_maps   

                for gt_label in gt_labels:
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

        # new text forward
        text_dict = self.language_model(new_text_prompts)
        visual_features = self.extract_feat(batch_inputs)

        # text_dict = self.language_model(ori_text_prompts)
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
                        
        # chunked new class text token mask for full class prompt
        if self.new_text_token_mask_chunked is None:
            new_text_token_mask_chunk_pos = self.token_positive_maps[len(batch_data_samples[0].ori_text)+1][0]
            new_text_token_mask_chunked = copy.deepcopy(text_token_mask)  
            new_text_token_mask_chunked[:new_text_token_mask_chunk_pos] = False
            new_text_token_mask_chunked = new_text_token_mask_chunked.unsqueeze(0).repeat(len(batch_data_samples),1)
            self.new_text_masks = new_text_token_mask_chunked   # text mask for new class  
            self.ori_text_masks = ~new_text_token_mask_chunked
        
        # switch to expert_0 in group and forward for distillation
        with torch.no_grad():
            self.eval()
            # ori model forward on full text
            if self.distn_cfg.future_class:
                ori_text_dict = text_dict
            else:
                ori_text_dict = self.language_model(ori_text_prompts)

            ori_visual_features = self.extract_feat(batch_inputs)
            ori_text_dict['embedded'] = self.text_feat_map(ori_text_dict['embedded'])
            # ori_visual_features = self.ori_model.extract_feat(batch_inputs)

            ori_head_inputs_dict = self.forward_with_expert_switch(ori_visual_features, ori_text_dict, batch_data_samples)
            all_layers_ori_cls_scores, all_layers_ori_bbox_preds = \
                self.bbox_head(ori_head_inputs_dict['ori_hidden_states'], 
                                ori_head_inputs_dict['ori_references'], 
                                ori_head_inputs_dict['ori_memory_text'], 
                                ori_head_inputs_dict['ori_text_token_mask'])
            ori_head_inputs_dict['all_layers_ori_cls_scores'] = all_layers_ori_cls_scores
            ori_head_inputs_dict['all_layers_ori_bbox_preds'] = all_layers_ori_bbox_preds
            ori_head_inputs_dict['ori_token_positive_maps'] = self.ori_token_positive_maps

            if self.distn_cfg.future_class:
                ori_text_len = self.ori_token_positive_maps[len(data_samples.ori_text)][-1] + 1
                ori_text_token_mask =  ori_head_inputs_dict['ori_text_token_mask'][:,:ori_text_len]
                ori_head_inputs_dict['ori_text_token_mask'] = ori_text_token_mask
            else:
                ori_text_token_mask = ori_head_inputs_dict['ori_text_token_mask']

            if self.distn_cfg.label_distn.type == 'topk_pseudo' or self.distn_cfg.label_distn.type == 'threshold_pseudo':
                topk_query, batch_pseudo_instances, batch_all_instances = \
                    self.bbox_head.generate_pseudo_label(all_layers_ori_cls_scores,
                                                        all_layers_ori_bbox_preds,
                                                        ori_text_token_mask,
                                                        text_token_mask, 
                                                        batch_data_samples,
                                                        self.ori_token_positive_maps)    
                
            ori_head_inputs_dict['batch_pseudo_instances'] = batch_pseudo_instances
            ori_head_inputs_dict['batch_all_instances'] = batch_all_instances
            ori_head_inputs_dict['ori_topk_query'] = topk_query
            self.train()
        
        aux_dict = None
        if self.distn_cfg.query_distn.type == 'seperate_queryinit':
            num_distn_queries = self.distn_cfg.query_distn.num_aux_query
            assert num_distn_queries <= self.distn_cfg.query_distn.num_matching_query
            aux_query = ori_head_inputs_dict['aux_query'].clone()
            aux_reference = ori_head_inputs_dict.pop('aux_reference')
            aux_query = aux_query[:, :num_distn_queries]
            aux_reference = aux_reference[:, :num_distn_queries]
            aux_enc_coord = ori_head_inputs_dict['enc_outputs_coord'].clone() 
            aux_enc_score = ori_head_inputs_dict['enc_outputs_class'].clone()

            aux_dict = dict(aux_query=aux_query, aux_enc_coord=aux_enc_coord, aux_enc_score=aux_enc_score, 
                            aux_reference=aux_reference, batch_pseudo_instances=batch_pseudo_instances)
        
        new_head_inputs_dict, old_head_inputs_dict = self.forward_transformer(visual_features,  text_dict, 
                                                                              batch_data_samples, aux_dict)    
        # new_head_inputs_dict['text_token_mask_chunked'] = new_text_token_mask_chunked    
        new_head_inputs_dict['token_positive_maps'] = self.token_positive_maps 
        
        if 'dn_meta' in ori_head_inputs_dict.keys():
            ori_head_inputs_dict.pop('dn_meta')
        # if 'enc_outputs_class' in ori_head_inputs_dict.keys():
        #     ori_head_inputs_dict.pop('enc_outputs_class')
        #     ori_head_inputs_dict.pop('enc_outputs_coord')
        
        losses = self.bbox_head.loss(new_head_inputs_dict, 
                                     old_head_inputs_dict, 
                                     ori_head_inputs_dict, 
                                     batch_data_samples=batch_data_samples)
        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
        for i_ds, data_samples in enumerate(batch_data_samples):
            # teacher mode for pseudo labeling: use ori_text (old classes) as prompt
            if self.vis_cfg.type == 'pseudo_labeling' and \
               hasattr(data_samples, 'ori_text') and data_samples.ori_text is not None:
                text_prompts.append(data_samples.ori_text)
            else:
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

        if self.vis_cfg.type == 'pseudo_labeling' and hasattr(self, 'set_switch_signal'):
            task_id = self.task_id
            self.set_switch_signal(True)

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
