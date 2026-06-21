"""
extract_coco_features.py

Extract GDINO neck features from COCO (train2017 or val2017) as reference samples.
Saves each image as a [num_levels, C] .pt file to --save-dir.
"""

import torch
import os
import random
import argparse
from tqdm import tqdm
from mmengine.config import Config
from mmdet.registry import MODELS, DATASETS
from mmdet.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='Extract GDINO features from COCO for OOD evaluation')
    parser.add_argument('--checkpoint', default='weights/groundingdino_swint_ogc_mmdet-822d7e9d.pth')
    parser.add_argument('--coco-root', default='/data/wangxu/coco')
    parser.add_argument('--split', default='val2017', choices=['train2017', 'val2017'])
    parser.add_argument('--task-name', default=None, help='Task name (e.g. coco_40-49), overrides --save-dir logic if set')
    parser.add_argument('--save-dir', default=None)
    parser.add_argument('--max-samples', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    register_all_modules(init_default_scope=True)

    if args.save_dir is None:
        if args.task_name:
            args.save_dir = f'visualize/CDIOD_mlvlfeat/{args.task_name}'
        else:
            args.save_dir = f'visualize/CDIOD_mlvlfeat/COCO_{args.split.split("2017")[0]}'

    os.makedirs(args.save_dir, exist_ok=True)
    existing = [f for f in os.listdir(args.save_dir) if f.endswith('.pt')]
    if len(existing) >= args.max_samples:
        print(f"Already extracted {len(existing)} COCO features. Skipping.")
        return

    # Build model using group_moe_extract_feats config (all frozen, no domain predictor)
    cfg = Config.fromfile('projects/DGS/configs/CDIOD/extract_feats.py')
    cfg.model.num_tasks = 1
    cfg.model.task_id = 0
    if 'domain_predictor_cfg' in cfg.model:
        cfg.model.domain_predictor_cfg = None

    print("Building GDINO model...")
    model = MODELS.build(cfg.model)
    from mmengine.runner import load_checkpoint
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    # Build COCO dataset based on split
    ann_file = f'annotations/instances_{args.split}.json'
    data_prefix = dict(img=f'{args.split}/')
    
    coco_ds_cfg = dict(
        type='CocoDataset',
        data_root=args.coco_root + '/',
        ann_file=ann_file,
        data_prefix=data_prefix,
        test_mode=True,
        pipeline=cfg.test_dataloader.dataset.pipeline,
        return_classes=True,
        backend_args=None,
    )
    print(f"Building COCO {args.split} dataset...")
    dataset = DATASETS.build(coco_ds_cfg)
    print(f"Total COCO images in {args.split}: {len(dataset)}")

    # Random sample fixed subset
    random.seed(args.seed)
    indices = list(range(len(dataset)))
    if len(indices) > args.max_samples:
        indices = random.sample(indices, args.max_samples)
    indices.sort()
    print(f"Extracting {len(indices)} COCO features to {args.save_dir}...")

    with torch.no_grad():
        for idx in tqdm(indices):
            data_item = dataset[idx]
            inputs = data_item['inputs']
            data_samples = [data_item['data_samples']]

            if hasattr(model, 'data_preprocessor'):
                batch_data = model.data_preprocessor(
                    {'inputs': [inputs], 'data_samples': data_samples}, training=False)
                inputs = batch_data['inputs']
            else:
                inputs = inputs.unsqueeze(0).float()
                if torch.cuda.is_available():
                    inputs = inputs.cuda()

            visual_features = model.extract_feat(inputs)

            lvl_feats = []
            for feat in visual_features:
                pooled = torch.nn.functional.adaptive_avg_pool2d(feat, 1)
                lvl_feats.append(pooled.view(-1))
            stacked_feat = torch.stack(lvl_feats, dim=0)  # [num_levels, C]

            img_id = data_samples[0].img_id
            save_path = os.path.join(args.save_dir, f"img_{img_id}.pt")
            torch.save(stacked_feat.cpu(), save_path)

    final_cnt = len([f for f in os.listdir(args.save_dir) if f.endswith('.pt')])
    print(f"Done! {final_cnt} COCO features saved to {args.save_dir}")


if __name__ == '__main__':
    main()
