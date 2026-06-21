import torch
import os
import argparse
from tqdm import tqdm
from mmengine.config import Config
from mmdet.registry import MODELS, DATASETS
from mmdet.utils import register_all_modules

def parse_args():
    parser = argparse.ArgumentParser(description='Extract GDINO neck features for ODinW13 datasets')
    parser.add_argument('--config', default='projects/DGS/configs/IVLOD/extract_feats.py')
    parser.add_argument('--checkpoint', default='weights/groundingdino_swint_ogc_mmdet-822d7e9d.pth')
    parser.add_argument('--save-dir', default='visualize/CDIOD_mlvlfeat/ODinW13')
    parser.add_argument('--data-root', default='/data/wangxu/ODinW13')
    parser.add_argument('--split', default='train', choices=['train', 'valid', 'test'],
                        help='Dataset split to extract features from')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples per task (default: all)')
    return parser.parse_args()

def main():
    args = parse_args()
    register_all_modules(init_default_scope=True)

    # ODinW13 tasks
    datasets_names = ('AerialMaritimeDrone', 'Aquarium', 'CottontailRabbits', 'EgoHands', 'NorthAmericaMushroom',
                     'Packages', 'PascalVOC', 'pistols', 'pothole', 'Raccoon',
                     'ShellfishOpenImages', 'thermalDogsAndPeople', 'VehiclesOpenImages')

    cfg = Config.fromfile(args.config)

    # Inject task parameters to satisfy MoE layer requirements during build
    cfg.model.num_tasks = 1
    cfg.model.task_id = 0

    # Ensure model is in eval mode and doesn't need DGS components for pure extraction
    if 'domain_predictor_cfg' in cfg.model:
        cfg.model.domain_predictor_cfg = None

    print(f"Building GDINO model...")
    model = MODELS.build(cfg.model)
    if args.checkpoint:
        from mmengine.runner import load_checkpoint
        load_checkpoint(model, args.checkpoint, map_location='cpu')

    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    for task_name in datasets_names:
        # Save to {save_dir}/{task_name}/{split}/
        task_save_dir = os.path.join(args.save_dir, task_name, args.split)
        
        # Check if already processed (this handles re-runs by skipping completed tasks)
        # Note: We now don't skip based on a small count, but we keep the directory check 
        # to allow resuming if the user stops the script.
        # Use a higher threshold or just check if it exists at all.
        if os.path.exists(task_save_dir) and len(os.listdir(task_save_dir)) > 0:
             print(f"Directory {task_save_dir} exists and contains files. We will overwrite or append (if needed).")
             # Actually, if we want to extract ALL, we should probably check if it matches the dataset size.
             # But for simplicity, we'll just proceed or the user can manual delete.
             
        os.makedirs(task_save_dir, exist_ok=True)
        
        print(f"\n" + "="*40)
        print(f"Processing Task: {task_name} [{args.split}]")

        task_data_root = os.path.join(args.data_root, task_name)
        ann_file = f"{args.split}/annotations_without_background.json"

        # Build dataset config directly (group_moe_extract_feats uses single dataset)
        ds_cfg = cfg.test_dataloader.dataset.copy()
        # Handle both single dataset and nested datasets list
        if hasattr(ds_cfg, 'datasets'):
            ds_cfg = ds_cfg.datasets[0]
        ds_cfg.data_root = task_data_root + '/'
        ds_cfg.metainfo = task_name
        ds_cfg.ann_file = ann_file
        ds_cfg.data_prefix = dict(img=f'{args.split}/')

        print(f"Building dataset for {task_name} [{args.split}]...")
        dataset = DATASETS.build(ds_cfg)
        num_samples = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)

        print(f"Extracting features to {task_save_dir}...")
        with torch.no_grad():
            for i in tqdm(range(num_samples)):
                data_item = dataset[i]
                inputs = data_item['inputs']
                data_samples = [data_item['data_samples']]
                
                # Preprocess data
                if hasattr(model, 'data_preprocessor'):
                    batch_data = model.data_preprocessor({'inputs': [inputs], 'data_samples': data_samples}, training=False)
                    inputs = batch_data['inputs']
                else:
                    inputs = inputs.unsqueeze(0).float()
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()

                # Extract features (Backbone -> Neck)
                visual_features = model.extract_feat(inputs) # tuple of levels
                
                # Apply Global Average Pooling to each level
                lvl_feats = []
                for feat in visual_features:
                    pooled = torch.nn.functional.adaptive_avg_pool2d(feat, 1)
                    lvl_feats.append(pooled.view(-1)) # [C]
                
                # Stack to [num_levels, C]
                stacked_feat = torch.stack(lvl_feats, dim=0)
                
                # Save feature
                img_id = data_samples[0].img_id
                save_path = os.path.join(task_save_dir, f"img_{img_id}.pt")
                
                # Move to CPU for saving
                torch.save(stacked_feat.cpu(), save_path)

        print(f"Done! Extracted {num_samples} features for {task_name} to {task_save_dir}")

    print(f"\nAll ODinW13 tasks completed!")

if __name__ == '__main__':
    main()
