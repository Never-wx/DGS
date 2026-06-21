_base_ = '../_base_/gdino_inc_agnostic_CDIOD.py'
custom_imports = dict(imports=["projects.DGS.dgs"], allow_failed_imports=False)
caption_prompt = None

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        _delete_=True,
        type='CDIOD_Agnostic_Dataset',
        metainfo='',
        data_root='',
        ann_file='',
        data_prefix=dict(img='train/'),
        pipeline=_base_.test_pipeline,
        caption_prompt=caption_prompt,
        test_mode=True,
        return_classes=True,
        )
    )

val_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    ann_file='',
    metric=['bbox'],
    format_only=False,
    backend_args=None)

test_dataloader = val_dataloader
test_evaluator = val_evaluator


