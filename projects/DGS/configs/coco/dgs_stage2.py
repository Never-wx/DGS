_base_ = './dgs_stage1.py'

distn_cfg=dict(
    type='distillation',
    future_class=False,
    label_distn=dict(
        type='threshold_pseudo', # choice = ['topk_pseudo', 'threshold_pseudo', 'adaptive_pseudo']
        mode='hardlabel', # choice = ['seperate_logits', 'hardlabel', 'mixed_logits']
        sigma=0.4, label_iou_th=0.7,
    ),
    feat_distn=dict(
        type='inter-intra', # choice = ['opt1', 'foreground', 'pseudo guided', 'inter-intra'']
        subtype = 'opt1', # choice = ['inter-text', 'inter-query', 'intra', 'q2t_distance', 'all']
        img_loss=dict(type='L2Loss', loss_weight=3.0, reduction='mean'),
        text_loss=dict(type='L2Loss', loss_weight=5.0, reduction='mean'),
    ),        
    query_distn=dict(
        type='seperate_queryinit',  # choice = [seperate_queryinit, balanced_queryinit, pseudo_denoising]
        num_matching_query=900,   
        num_aux_query=900,
    ), 
)

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
)

model = dict(
    type='GroundingDINO_DGS',
    num_queries=900,
    bbox_head=dict(
        type='GroundingDINOHead_inc_DGS',
        contrastive_cfg=dict(max_text_len=256, log_scale=None, bias=None),
        distn_cfg=distn_cfg),
    dn_cfg=None,
    distn_cfg=distn_cfg,
)

optim_wrapper = dict(optimizer=dict(type='AdamW', lr=0.0005, weight_decay=0.0001))

max_epochs = 12

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[11],
        gamma=0.1)
]

custom_hooks = [
    dict(type='WeightsTransformHook', 
         cfg=[dict(type='moe_lora'), 
              dict(type='moe_group_init')]
        ),
    dict(type='MergeHook', cfg=dict(type='ema')),  
    dict(type='DomainPredictorHooK')
]
