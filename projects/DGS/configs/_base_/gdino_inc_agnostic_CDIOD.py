_base_ = [
    '../../../../configs/_base_/datasets/coco_detection.py',
    '../../../../configs/_base_/schedules/schedule_1x.py', '../../../../configs/_base_/default_runtime.py',
]
lang_model_name = '/home/wangxu/.cache/huggingface/hub/bert-base-uncased'

model = dict(
    type='GroundingDINO_inc',
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=False,
    ),
    language_model=dict(
        type='BertModel',
        name=lang_model_name, 
        pad_to_max=False,
        use_sub_sentence_represent=True,
        special_tokens_list=['[CLS]', '[SEP]', '.', '?'],
        add_pooling_layer=False,
    ),
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=True,
        frozen_stages=-1,
        convert_weights=False),
    neck=dict(
        type='ChannelMapper',
        in_channels=[192, 384, 768],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        bias=True,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        num_layers=6,
        num_cp=6,
        # visual layer config
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        # text layer config
        text_layer_cfg=dict(
            self_attn_cfg=dict(num_heads=4, embed_dims=256, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.0)),
        # fusion layer config
        fusion_layer_cfg=dict(
            v_dim=256,
            l_dim=256,
            embed_dim=1024,
            num_heads=4,
            init_values=1e-4),
    ),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            # query self attention layer
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # cross attention layer query to text
            cross_attn_text_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # cross attention layer query to image
            cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(
        type='GroundingDINOHead_inc',
        setting='cur_text',
        # num_classes=80,
        trunc_class=[0, 256],
        sync_cls_avg_factor=True,
        contrastive_cfg=dict(max_text_len=256, log_scale=None, bias=None),
        # contrastive_cfg=dict(max_text_len=256, log_scale=0.0, bias=None),
        # contrastive_cfg=dict(max_text_len=256, log_scale='auto', bias=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='BinaryFocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    # test_cfg=dict(max_per_img=300, chunked_size=2),
    test_cfg=dict(max_per_img=300),
    frozen_cfg=dict(
        backbone_frozen=False,
        language_model_frozen=True,
        neck_frozen=False,
        encoder_frozen=False,
        decoder_frozen=False,
        head_frozen=False)
    )

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'ori_text', 'custom_entities'))
]

test_pipeline = [
    dict(
        type='LoadImageFromFile', backend_args=None,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'ori_text', 'custom_entities', 'caption_prompt'))
]

# -------------------------------------------------#
dataset_type = 'CDIOD_Agnostic_Dataset'
data_root = 'data/CDIOD/'
data_metric = 'CDIODMetric'
ann_file_path = 'valid/annotations_without_background.json'
base_test_pipeline = _base_.test_pipeline
base_test_pipeline[-1]['meta_keys'] = ('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'text',
                                       'custom_entities', 'caption_prompt')

caption_prompt = None
use_mp_eval=False

# metainfo = ('animals', 'cityscape', 'clipart1k', 'construction_safety' ,'DIOR', 
#             'document_parts', 'liver_disease', 'NEU-DET', 'RUOD', 'washroom')
# metainfo = ('animals', 'clipart1k', 'construction_safety')
metainfo = ('DIOR', 'PascalVOC', 'RUOD')

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        # type='CocoIncDataset',
        type=dataset_type,
        data_prefix=dict(img='train/'),
        # ann_file='train/annotations_without_background.json',
        ann_file='',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
        return_classes=True))

datasets = []
metrics = []

dataset_prefixes = list(metainfo)

for prefix in dataset_prefixes:
    
    _data_root = data_root + f'{prefix}/'
    dataset = dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=_data_root,
        ann_file=ann_file_path,
        data_prefix=dict(img='valid/'),
        pipeline=base_test_pipeline,
        caption_prompt=caption_prompt,
        test_mode=True,
        return_classes=True
    )
    
    metric = dict(
        type=data_metric,
        ann_file=_data_root + ann_file_path,
        metric='bbox',
        # iou_thrs=[0.5],
        metric_items=['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l', 'AR@100', 'AR@300'],
        use_mp_eval=use_mp_eval
    )
    
    datasets.append(dataset)
    metrics.append(metric)

val_dataloader = dict(
    dataset=dict(_delete_=True, type='ConcatDataset', datasets=datasets))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='CDIODMultiDatasetsEvaluator',
    metrics=metrics,
    dataset_prefixes=dataset_prefixes,
    evaluation_tasks='')

test_evaluator = val_evaluator
# -------------------------------------------------#

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    # optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    optimizer=dict(type='AdamW', lr=0.00005, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            # 'language_model': dict(lr_mult=0.0),   # open set continual finetune
            'language_model': dict(lr_mult=0.1), # close set finetune
        }))
# learning policy
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

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (4 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)

default_hooks = dict(
    checkpoint=dict(save_optimizer=False, save_param_scheduler=False, max_keep_ckpts=1),
    visualization=dict(type='GroundingVisualizationHook'))