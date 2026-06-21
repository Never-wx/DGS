import torch
import os
import argparse
from tqdm import tqdm
from mmengine.config import Config
from mmdet.registry import MODELS, DATASETS
from mmdet.utils import register_all_modules

def parse_args():
    parser = argparse.ArgumentParser(description='Extract GDINO neck features for CDIOD tasks (Optimized)')
    parser.add_argument('--config', default='projects/DGS/configs/CDIOD/extract_feats.py')
    parser.add_argument('--checkpoint', default='weights/groundingdino_swint_ogc_mmdet-822d7e9d.pth')
    parser.add_argument('--save-dir', default='visualize/CDIOD_git/')
    parser.add_argument('--data-root', default='data/CDIOD/')
    parser.add_argument('--split', default='train', choices=['train', 'valid'],
                        help='Dataset split to extract features from (train or valid)')
    parser.add_argument('--steps', type=int, default=10, choices=[5, 10],
                        help='Task sequence steps (5 or 10)')
    parser.add_argument('--max-samples', type=int, default=1000000,
                        help='Max samples per task')
    return parser.parse_args()

def main():
    args = parse_args()
    register_all_modules(init_default_scope=True)

    # 1. Define task sequences (Borrowed from extract_glip_features layout for clarity)
    if args.steps == 10:
        tasks = [
            {'name': 'DIOR_Task4-1', 'path': 'DIOR', 'ann': f'{args.split}/annotations_without_background_Task4-1.json'},
            {'name': 'DIOR_Task4-2', 'path': 'DIOR', 'ann': f'{args.split}/annotations_without_background_Task4-2.json'},
            {'name': 'DIOR_Task4-3', 'path': 'DIOR', 'ann': f'{args.split}/annotations_without_background_Task4-3.json'},
            {'name': 'DIOR_Task4-4', 'path': 'DIOR', 'ann': f'{args.split}/annotations_without_background_Task4-4.json'},
            {'name': 'PascalVOC_Task4-1', 'path': 'PascalVOC', 'ann': f'{args.split}/annotations_without_background_Task4-1.json'},
            {'name': 'PascalVOC_Task4-2', 'path': 'PascalVOC', 'ann': f'{args.split}/annotations_without_background_Task4-2.json'},
            {'name': 'PascalVOC_Task4-3', 'path': 'PascalVOC', 'ann': f'{args.split}/annotations_without_background_Task4-3.json'},
            {'name': 'PascalVOC_Task4-4', 'path': 'PascalVOC', 'ann': f'{args.split}/annotations_without_background_Task4-4.json'},
            {'name': 'RUOD_Task2-1', 'path': 'RUOD', 'ann': f'{args.split}/annotations_without_background_Task2-1.json'},
            {'name': 'RUOD_Task2-2', 'path': 'RUOD', 'ann': f'{args.split}/annotations_without_background_Task2-2.json'},
        ]
    else: # steps == 5
        tasks = [
            {'name': 'DIOR_Task2-1', 'path': 'DIOR', 'ann': f'{args.split}/annotations_without_background_Task2-1.json'},
            {'name': 'DIOR_Task2-2', 'path': 'DIOR', 'ann': f'{args.split}/annotations_without_background_Task2-2.json'},
            {'name': 'PascalVOC_Task2-1', 'path': 'PascalVOC', 'ann': f'{args.split}/annotations_without_background_Task2-1.json'},
            {'name': 'PascalVOC_Task2-2', 'path': 'PascalVOC', 'ann': f'{args.split}/annotations_without_background_Task2-2.json'},
            {'name': 'RUOD', 'path': 'RUOD', 'ann': f'{args.split}/annotations_without_background.json'},
        ]

    # 2. Build Model
    cfg = Config.fromfile(args.config)
    
    # Ensure raw neck features are extracted without extra DGS logic
    if 'domain_predictor_cfg' in cfg.model:
        cfg.model.domain_predictor_cfg = None
    if 'moe_cfg' in cfg.model:
        cfg.model.moe_cfg = None

    print(f"Building detector (Backbone + Neck)...")
    model = MODELS.build(cfg.model)
    if args.checkpoint:
        from mmengine.runner import load_checkpoint
        load_checkpoint(model, args.checkpoint, map_location='cpu')

    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    # 3. Extraction Loop
    for task in tasks:
        task_name = task['name']
        # features are saved to: {save-dir}/{task_name}/{split}/
        task_save_dir = os.path.join(args.save_dir, task_name, args.split)
        os.makedirs(task_save_dir, exist_ok=True)
        
        print(f"\n" + "="*40)
        print(f"Processing Task: {task_name} (Base: {base_dataset}) [{args.split}]")

        # Update dataset config using GLIP-style explicit assignment
        ds_cfg = cfg.test_dataloader.dataset.copy()
        if hasattr(ds_cfg, 'datasets'): # Handle multi-dataset wrappers if present
            ds_cfg = ds_cfg.datasets[0]
            
        ds_cfg.data_root = os.path.join(args.data_root, task['path']) + '/'
        ds_cfg.ann_file = task['ann']
        ds_cfg.metainfo = task_name # CDIOD_Agnostic_Dataset lookups this in its METAINFO
        ds_cfg.data_prefix = dict(img=f'{args.split}/')
        
        # Ensure extraction-only mode
        ds_cfg.test_mode = True
        if 'memory_cfg' in ds_cfg: ds_cfg.memory_cfg = None
        if 'distn_cfg' in ds_cfg: ds_cfg.distn_cfg = None

        print(f"Building dataset...")
        dataset = DATASETS.build(ds_cfg)
        num_samples = min(len(dataset), args.max_samples)

        print(f"Extracting features to {task_save_dir} (Skip Existing)...")
        num_skipped = 0
        with torch.no_grad():
            for i in tqdm(range(num_samples)):
                data_item = dataset[i]
                data_samples = [data_item['data_samples']]
                img_id = data_samples[0].img_id
                save_path = os.path.join(task_save_dir, f"img_{img_id}.pt")
                
                # Persistence: Skip existing to save time/compute
                if os.path.exists(save_path):
                    num_skipped += 1
                    continue

                inputs = data_item['inputs']
                if hasattr(model, 'data_preprocessor'):
                    batch_data = model.data_preprocessor({'inputs': [inputs], 'data_samples': data_samples}, training=False)
                    inputs = batch_data['inputs']
                else:
                    inputs = inputs.unsqueeze(0).float()
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()

                # Extract multi-level neck features
                visual_features = model.extract_feat(inputs) 
                
                # Pool each level to [C]
                lvl_feats = []
                for feat in visual_features:
                    pooled = torch.nn.functional.adaptive_avg_pool2d(feat, 1)
                    lvl_feats.append(pooled.view(-1))
                
                # Stack to [num_levels, C]
                stacked_feat = torch.stack(lvl_feats, dim=0)
                torch.save(stacked_feat.cpu(), save_path)

        print(f"Done! Extracted: {num_samples - num_skipped}, Skipped: {num_skipped}")

    print(f"\nAll tasks in {args.steps}-step sequence completed!")

if __name__ == '__main__':
    main()
