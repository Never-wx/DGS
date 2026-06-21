_base_ = '../_base_/gdino_inc_agnostic_CDIOD.py'
custom_imports = dict(imports=["projects.DGS.dgs"], allow_failed_imports=False)

# load_from = './weights/groundingdino_swint_ogc_mmdet-822d7e9d.pth' 
caption_prompt = None

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
)

test_dataloader = val_dataloader

moe_cfg = dict(type='moe_adaptive_expand_lora', experts_num=1, top_k=1, 
               r=16, alpha=32.0, dropout=0.0, 
               group_cfg=dict(type='rep', merge_method='ema', lambda_A=0.2, lambda_B=0.2),
               replace_layer_type=['enc_ffn_img', 'enc_ffn_text'],
               replace_enc_layer_ids = [0,1,2,3,4,5], replace_dec_layer_ids=[]) 

vis_cfg = dict(type='none', save_path='')

model = dict(
    num_queries=900,
    type='GroundingDINO_DGS_Base',
    encoder=dict(
        layer_cfg=dict(
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        text_layer_cfg=dict(
            ffn_cfg=dict(embed_dims=256, feedforward_channels=1024, ffn_drop=0.0)),
        fusion_layer_cfg=dict(
            v_dim=256, l_dim=256, embed_dim=1024, num_heads=4, init_values=1e-4)
    ),
    frozen_cfg=dict(
        backbone_frozen=True,
        language_model_frozen=True,
        neck_frozen=True,
        encoder_frozen=True,
        decoder_frozen=True,
        head_frozen=True,
        exclude_keywords = ['lora_']
    ),
    moe_cfg=moe_cfg,
    vis_cfg=vis_cfg,
    domain_predictor_cfg=dict(
        type='svd',
        feat_path='visualize/CDIOD_mlvlfeat/',
        stats_path='work_dirs/CDIOD/feature_stats/svd',
        multilevel=False,
        expand_th=150,
        ood_th=500,
        min_eig_ratio=1e-3,
    )
)

optim_wrapper = dict(optimizer=dict(type='AdamW', lr=0.0008, weight_decay=0.0001))

max_epochs = 12

train_cfg = dict(max_epochs=max_epochs, val_interval=max_epochs)

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

