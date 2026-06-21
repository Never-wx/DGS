_base_ = '../_base_/gdino_inc_ODinW13.py'
caption_prompt = None

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _delete_=True,
        metainfo = '',
        data_root='',
        type='ODinW13_Dataset',
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




