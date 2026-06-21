import torch
import torch.nn.functional as F
from mmdet.models.dense_heads.atss_vlfusion_head import convert_grounding_to_cls_scores
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, bbox_overlaps, bbox2roi
from mmdet.models.utils import multi_apply
from mmengine.structures import InstanceData

def pdist(e, dist_mode="l2", eps=1e-6):
    dist_mode = dist_mode.lower()
    assert dist_mode in ["l1", "l2"], dist_mode
    N = e.shape[0]
    if dist_mode == "l1":
        res = torch.abs(e.unsqueeze(1) - e.unsqueeze(0)).clamp(min=eps)
    elif dist_mode == 'l2':
        e_square = e.pow(2).sum(dim=1)
        prod = torch.matmul(e, e.T)
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
    else:
        raise NotImplementedError  
    
    res = res.clone()
    res[range(N), range(N)] = 0
    mean_res = (res[res>0].mean()).clamp(min=eps)
    res_norm = res / mean_res
    return res_norm

def inter_text_relation(ori_token_positive_maps, text_feats, ori_text_feats):
    # text_prototype
    text_prototype_percls_list = []
    ori_text_prototype_percls_list = []
    for k, pos in ori_token_positive_maps.items():
        batch_text_i = text_feats[:, pos]
        batch_ori_text_i = ori_text_feats[:, pos]
        batch_text_i = torch.mean(batch_text_i, dim=1)    # [B,256]
        batch_ori_text_i = torch.mean(batch_ori_text_i, dim=1)    # [B,256]
        text_prototype = torch.mean(batch_text_i, dim=0)    # [1, 256]
        ori_text_prototype = torch.mean(batch_ori_text_i, dim=0)    
        text_prototype_percls_list.append(text_prototype)     # [num_cls, 256]
        ori_text_prototype_percls_list.append(ori_text_prototype)
    text_prototype_percls = torch.stack(text_prototype_percls_list, dim=0)
    ori_text_prototype_percls = torch.stack(ori_text_prototype_percls_list, dim=0)     

    ### interclass global text relation loss
    norm_text_diff_matrix = pdist(text_prototype_percls, dist_mode='l2')
    ori_norm_text_diff_matrix = pdist(ori_text_prototype_percls, dist_mode='l2')  
    # text_relation_loss = self.loss_textfeat_kd(norm_text_diff_matrix, ori_norm_text_diff_matrix)  
    return text_prototype_percls, ori_text_prototype_percls, norm_text_diff_matrix, ori_norm_text_diff_matrix

def inter_text_relation_partial(token_positive_maps, ori_token_positive_maps,
                                unique_pseudo_labels, ori_pseudo_labels_list, 
                                text_feats, ori_text_feats,
                                batch_cls_scores, batch_ori_cls_scores,
                                batch_query_feats, weighted=True):
    
    B,N,D = batch_query_feats.size()       
    if unique_pseudo_labels.size(0)>0:            
        total_cls_scores = batch_cls_scores.reshape(-1, D)    # [B*N, D]
        ori_total_cls_scores = batch_ori_cls_scores.reshape(-1, D)      
        total_output_score = convert_grounding_to_cls_scores(logits=total_cls_scores.sigmoid()[None],
                                                            positive_maps=[token_positive_maps])[0]       
        ori_total_output_score = convert_grounding_to_cls_scores(logits=ori_total_cls_scores.sigmoid()[None],
                                                                positive_maps=[ori_token_positive_maps])[0]                 
        total_weights, total_labels = torch.max(total_output_score, dim=1)
        ori_total_weights, ori_total_labels = torch.max(ori_total_output_score, dim=1)
        
        text_prototype_percls_list = []
        ori_text_prototype_percls_list = []
        for cls in unique_pseudo_labels:
            idx_cls = (total_labels==cls)
            ori_idx_cls = (ori_total_labels==cls)
            if torch.any(idx_cls):
                text_pos = ori_token_positive_maps[cls.item()+1]
                text_cls = []
                ori_text_cls = []
                # only collect text in current img
                for i, cur_cls in enumerate(ori_pseudo_labels_list):
                    if cls in cur_cls:
                        mean_text = torch.mean(text_feats[i, text_pos], dim=0)  # [256]
                        mean_ori_text = torch.mean(ori_text_feats[i, text_pos], dim=0)  # [256]
                        text_cls.append(mean_text)
                        ori_text_cls.append(mean_ori_text)
                text_cls = torch.stack(text_cls, dim=0)  # [n_cls, 256]
                ori_text_cls = torch.stack(ori_text_cls, dim=0)                   
                text_prototype = torch.mean(text_cls, dim=0)    # [256]
                ori_text_prototype = torch.mean(ori_text_cls, dim=0)    
                text_prototype_percls_list.append(text_prototype)     # num_cls * [256]
                ori_text_prototype_percls_list.append(ori_text_prototype)  
        
    ### interclass global text relation loss
    if unique_pseudo_labels.size(0)>2 and len(text_prototype_percls_list) > 2:
        text_prototype_percls = torch.stack(text_prototype_percls_list, dim=0)
        ori_text_prototype_percls = torch.stack(ori_text_prototype_percls_list, dim=0)    
        norm_text_diff_matrix = pdist(text_prototype_percls, dist_mode='l2')
        ori_norm_text_diff_matrix = pdist(ori_text_prototype_percls, dist_mode='l2')  
        # text_relation_loss = self.loss_textfeat_kd(norm_text_diff_matrix, ori_norm_text_diff_matrix)  
    else:
        text_prototype_percls = []
        ori_text_prototype_percls = []
        norm_text_diff_matrix = []
        ori_norm_text_diff_matrix = []

    return text_prototype_percls, ori_text_prototype_percls, norm_text_diff_matrix, ori_norm_text_diff_matrix

def inter_query_relation(token_positive_maps, ori_token_positive_maps, unique_pseudo_labels, 
                         batch_cls_scores, batch_ori_cls_scores,
                         batch_query_feats, batch_ori_query_feats, weighted=True):
    
    assert batch_query_feats.size() == batch_ori_query_feats.size()
    B, N, D = batch_query_feats.size()  
    if unique_pseudo_labels.size(0) > 0:            
        total_query_feats =  batch_query_feats.reshape(-1, D) # [B*N, 256]
        total_cls_scores = batch_cls_scores.reshape(-1, D)    # [B*N, D]
        ori_total_query_feats = batch_ori_query_feats.reshape(-1, D)
        ori_total_cls_scores = batch_ori_cls_scores.reshape(-1, D)      

        total_output_score = convert_grounding_to_cls_scores(logits=total_cls_scores.sigmoid()[None],
                                                            positive_maps=[token_positive_maps])[0]       
        ori_total_output_score = convert_grounding_to_cls_scores(logits=ori_total_cls_scores.sigmoid()[None],
                                                                positive_maps=[ori_token_positive_maps])[0]                 
        total_weights, total_labels = torch.max(total_output_score, dim=1)
        ori_total_weights, ori_total_labels = torch.max(ori_total_output_score, dim=1)
        
        # query_feats_percls_list = []
        query_prototype_percls_list = []
        # ori_query_feats_percls_list = []
        ori_query_prototype_percls_list = []
        for cls in unique_pseudo_labels:
            idx_cls = (total_labels==cls)
            ori_idx_cls = (ori_total_labels==cls)
            if torch.any(idx_cls):
                query_feats_cls = total_query_feats[idx_cls]
                ori_query_feats_cls = ori_total_query_feats[ori_idx_cls]
                if weighted:
                    weights = (total_weights[idx_cls] / total_weights[idx_cls].sum()).unsqueeze(1)
                    ori_weights = (ori_total_weights[ori_idx_cls] / ori_total_weights[ori_idx_cls].sum()).unsqueeze(1)
                    query_prototype_cls = (weights * query_feats_cls).sum(dim=0)
                    ori_query_prototype_cls = (ori_weights * ori_query_feats_cls).sum(dim=0)
                else:
                    query_prototype_cls = torch.mean(query_feats_cls, dim=0)
                    ori_query_prototype_cls = torch.mean(ori_query_feats_cls, dim=0)
                query_prototype_percls_list.append(query_prototype_cls) 
                ori_query_prototype_percls_list.append(ori_query_prototype_cls)
        
    ## interclass query relation loss, relation matrix = I where class <=2 
    if unique_pseudo_labels.size(0) > 2 and len(query_prototype_percls_list) > 2:
        query_prototype_percls = torch.stack(query_prototype_percls_list, dim=0)
        ori_query_prototype_percls = torch.stack(ori_query_prototype_percls_list, dim=0)
        norm_query_diff_matrix = pdist(query_prototype_percls)
        ori_norm_query_diff_matrix = pdist(ori_query_prototype_percls)  
        # query_relation_loss = self.loss_imgfeat_kd(norm_query_diff_matrix, ori_norm_query_diff_matrix) 
    else:
        norm_query_diff_matrix = []
        ori_norm_query_diff_matrix = []
    return norm_query_diff_matrix, ori_norm_query_diff_matrix

def inter_relation_feat_distn(self, text_feats, ori_text_feats, 
                              enc_feat_dict, ori_enc_feat_dict, 
                              batch_query_feats, batch_ori_query_feats,
                              batch_cls_scores, batch_bbox_preds, 
                              batch_ori_cls_scores, batch_ori_bbox_preds, 
                              batch_pseudo_instances, batch_img_metas):
    
    loss_distn_imgfeat = None
    loss_distn_textfeat = None
    img_relation_loss = None
    query_relation_loss = None
    if 'text' in self.distn_cfg.feat_distn.subtype:
        # text_relation
        text_percls_list = []
        ori_text_percls_list = []
        for k, pos in self.ori_token_positive_maps.items():
            batch_text_i = text_feats[:, pos]
            batch_ori_text_i = ori_text_feats[:, pos]
            batch_text_i = torch.mean(batch_text_i, dim=1)    # [B,256]
            batch_ori_text_i = torch.mean(batch_ori_text_i, dim=1)    # [B,256]
            text_prototype = torch.mean(batch_text_i, dim=0)    # [1, 256]
            ori_text_prototype = torch.mean(batch_ori_text_i, dim=0)    
            text_percls_list.append(text_prototype)     # [num_cls, 256]
            ori_text_percls_list.append(ori_text_prototype)
        text_percls = torch.stack(text_percls_list, dim=0)
        ori_text_percls = torch.stack(ori_text_percls_list, dim=0)    
        norm_text_diff_matrix = pdist(text_percls)
        ori_norm_text_diff_matrix = pdist(ori_text_percls)  
        # diff_matrix = text_percls.unsqueeze(1) - text_percls.unsqueeze(0)
        # distance_matrix = torch.norm(diff_matrix, p=2, dim=2)     #  [num_cls, num_cls]        
        # ori_diff_matrix = ori_text_percls.unsqueeze(1) - ori_text_percls.unsqueeze(0)
        # ori_distance_matrix = torch.norm(ori_diff_matrix, p=2, dim=2)  
        loss_distn_textfeat = self.loss_textfeat_kd(norm_text_diff_matrix, ori_norm_text_diff_matrix)
        
    if 'query' in self.distn_cfg.feat_distn.subtype:
        B,N,D = batch_query_feats.size()    
        ori_pseudo_labels_list = []
        for pseudo_instance in batch_pseudo_instances:
            ori_pseudo_labels_list.append(pseudo_instance.labels)
        unique_pseudo_labels = torch.unique(torch.cat(ori_pseudo_labels_list, dim=0))
        if unique_pseudo_labels.size(0)>2:            
            # query feat
            total_query_feats =  batch_query_feats.reshape(B*N, D) # [B*N, 256]
            total_cls_scores = batch_cls_scores.reshape(B*N, D)    # [B*N, D]

            ori_total_query_feats = batch_ori_query_feats.reshape(B*N, D)
            ori_total_cls_scores = batch_ori_cls_scores.reshape(B*N, D)      

            total_output_score = convert_grounding_to_cls_scores(logits=total_cls_scores.sigmoid()[None],
                                                                positive_maps=[self.token_positive_maps])[0]       
            ori_total_output_score = convert_grounding_to_cls_scores(logits=ori_total_cls_scores.sigmoid()[None],
                                                                    positive_maps=[self.ori_token_positive_maps])[0]                 
            total_weights, total_labels = torch.max(total_output_score, dim=1)
            ori_total_weights, ori_total_labels = torch.max(ori_total_output_score, dim=1)
            
            query_percls = []
            ori_query_percls = []
            for cls in unique_pseudo_labels:
                idx_cls = (total_labels==cls)
                ori_idx_cls = (ori_total_labels==cls)
                if torch.any(idx_cls):
                    ## query prototype
                    query_feats_cls = total_query_feats[idx_cls]
                    query_prototype_cls = torch.mean(query_feats_cls, dim=0)
                    query_percls.append(query_prototype_cls)                    
                    ori_query_feats_cls = ori_total_query_feats[ori_idx_cls]
                    ori_query_prototype_cls = torch.mean(ori_query_feats_cls, dim=0)
                    ori_query_percls.append(ori_query_prototype_cls)
            ## query relation
            query_percls = torch.stack(query_percls, dim=0)
            ori_query_percls = torch.stack(ori_query_percls, dim=0)
            norm_query_diff_matrix = pdist(query_percls)
            ori_norm_query_diff_matrix = pdist(ori_query_percls)  
            query_relation_loss = self.loss_imgfeat_kd(norm_query_diff_matrix, ori_norm_query_diff_matrix)    
    
    if 'img-query' in self.distn_cfg.feat_distn.subtype:
        B,N,D = batch_query_feats.size()    
        ori_pseudo_labels_list = []
        for pseudo_instance in batch_pseudo_instances:
            ori_pseudo_labels_list.append(pseudo_instance.labels)
        unique_pseudo_labels = torch.unique(torch.cat(ori_pseudo_labels_list, dim=0))
        if unique_pseudo_labels.size(0)>2:
            # gather imgfeat per instance
            spatial_shapes = enc_feat_dict['spatial_shapes']
            level_start_index = enc_feat_dict['level_start_index']
            batch_enc_imgfeats = enc_feat_dict['memory']
            batch_ori_enc_imgfeats = ori_enc_feat_dict['ori_memory']  
            level_start_index = level_start_index.tolist()
            level_start_index.append(batch_ori_enc_imgfeats.size(1))     
            multilvl_img_feats = []     # lvl * [B, D, H_lvl, W_lvl]
            multilvl_ori_img_feats = []
            for lvl, hw in enumerate(spatial_shapes):
                H_lvl, W_lvl = hw
                level_idx_start = level_start_index[lvl]
                level_idx_end = level_start_index[lvl+1]                    
                lvl_batch_enc_imgfeats = batch_enc_imgfeats[:, level_idx_start:level_idx_end]   # [B, H_lvl*W_lvl, C]
                lvl_batch_ori_enc_imgfeats = batch_ori_enc_imgfeats[:, level_idx_start:level_idx_end]
                lvl_batch_enc_imgfeats = lvl_batch_enc_imgfeats.contiguous().view(B, H_lvl, W_lvl, D).permute(0, 3, 1, 2)
                lvl_batch_ori_enc_imgfeats = lvl_batch_ori_enc_imgfeats.contiguous().view(B, H_lvl, W_lvl, D).permute(0, 3, 1, 2)
                multilvl_img_feats.append(lvl_batch_enc_imgfeats)
                multilvl_ori_img_feats.append(lvl_batch_ori_enc_imgfeats)

            # convert proposal to bbox_feat
            batch_proposal_bbox = []
            batch_ori_proposal_bbox = []
            for i, (img_metas, bbox_preds, ori_bbox_preds) in enumerate(zip(batch_img_metas, batch_bbox_preds, 
                                                                            batch_ori_bbox_preds)):
                img_h, img_w = img_metas['img_shape']
                factor = batch_bbox_preds[0].new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
                bbox_preds = bbox_cxcywh_to_xyxy(bbox_preds) * factor
                ori_bbox_preds = bbox_cxcywh_to_xyxy(ori_bbox_preds) * factor
                batch_proposal_bbox.append(bbox_preds)
                batch_ori_proposal_bbox.append(ori_bbox_preds)
            batch_proposal_bbox = torch.stack(batch_proposal_bbox, dim=0)
            batch_ori_proposal_bbox = torch.stack(batch_ori_proposal_bbox, dim=0)         

            rois = bbox2roi(batch_proposal_bbox)   # [B,N,4] -> [B*N, 5]
            ori_rois = bbox2roi(batch_ori_proposal_bbox)   # [B,N,4] -> [B*N, 5] 
            total_bbox_feats = self.roi_extractor(multilvl_img_feats, rois)  # [B*N, D, out_h, out_w]
            total_ori_bbox_feats = self.roi_extractor(multilvl_ori_img_feats, ori_rois)

            # query feat
            total_query_feats =  batch_query_feats.reshape(B*N, D) # [B*N, 256]
            total_cls_scores = batch_cls_scores.reshape(B*N, D)    # [B*N, D]

            ori_total_query_feats = batch_ori_query_feats.reshape(B*N, D)
            ori_total_cls_scores = batch_ori_cls_scores.reshape(B*N, D)      

            total_output_score = convert_grounding_to_cls_scores(logits=total_cls_scores.sigmoid()[None],
                                                                positive_maps=[self.token_positive_maps])[0]       
            ori_total_output_score = convert_grounding_to_cls_scores(logits=ori_total_cls_scores.sigmoid()[None],
                                                                    positive_maps=[self.ori_token_positive_maps])[0]                 
            total_weights, total_labels = torch.max(total_output_score, dim=1)
            ori_total_weights, ori_total_labels = torch.max(ori_total_output_score, dim=1)
            
            query_percls = []
            ori_query_percls = []
            img_percls = []
            ori_img_percls = []
            for cls in unique_pseudo_labels:
                idx_cls = (total_labels==cls)
                ori_idx_cls = (ori_total_labels==cls)
                if torch.any(idx_cls):
                    ## img prototype
                    img_feats_cls = total_bbox_feats[idx_cls]
                    img_prototype_cls = torch.mean(img_feats_cls, dim=0)   
                    img_percls.append(img_prototype_cls) 
                    ori_img_feats_cls = total_ori_bbox_feats[ori_idx_cls]
                    ori_img_prototype_cls = torch.mean(ori_img_feats_cls, dim=0)
                    ori_img_percls.append(ori_img_prototype_cls)
                    ## query prototype
                    query_feats_cls = total_query_feats[idx_cls]
                    query_prototype_cls = torch.mean(query_feats_cls, dim=0)
                    query_percls.append(query_prototype_cls)                    
                    ori_query_feats_cls = ori_total_query_feats[ori_idx_cls]
                    ori_query_prototype_cls = torch.mean(ori_query_feats_cls, dim=0)
                    ori_query_percls.append(ori_query_prototype_cls)

            ## img relation
            img_percls = torch.stack(img_percls, dim=0)
            ori_img_percls = torch.stack(ori_img_percls, dim=0)
            norm_img_diff_matrix = pdist(img_percls.view(img_percls.size(0), -1))
            ori_norm_img_diff_matrix = pdist(ori_img_percls.view(ori_img_percls.size(0), -1))   
            ## query relation
            query_percls = torch.stack(query_percls, dim=0)
            ori_query_percls = torch.stack(ori_query_percls, dim=0)
            norm_query_diff_matrix = pdist(query_percls)
            ori_norm_query_diff_matrix = pdist(ori_query_percls)  
            img_relation_loss = self.loss_imgfeat_kd(norm_img_diff_matrix, ori_norm_img_diff_matrix)
            query_relation_loss = self.loss_imgfeat_kd(norm_query_diff_matrix, ori_norm_query_diff_matrix)               

    return img_relation_loss, query_relation_loss, loss_distn_textfeat