_base_ = './dgs_stage1.py'
base_test_pipeline = _base_.test_pipeline
dataset_type = 'CocoDataset'
data_root = './data/coco/'

model = dict(
    domain_predictor_cfg=dict(
        type='svd',
        feat_path='visualize/CDIOD_mlvlfeat/ODinW13_augment/',
        stats_path='work_dirs/ODinW13/feature_stats/svd',
        multilevel=False,
        expand_th=150,
        ood_th=200,
        min_eig_ratio=1e-3,
    )  
)

val_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=base_test_pipeline,
        return_classes=True,
        backend_args=None))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',
    format_only=False,
    backend_args=None)
test_evaluator = val_evaluator