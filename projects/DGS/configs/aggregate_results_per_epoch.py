import os
import json
import argparse
from collections import defaultdict

def aggregate_results_per_epoch(root_dir, output_file):
    """收集所有数据集的bbox_mAP变化数据"""
    results = defaultdict(list)
    
    # 遍历根目录下的所有数据集目录
    for dataset in os.listdir(root_dir):
        dataset_path = os.path.join(root_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue
        
        # 查找最新实验目录
        exp_dirs = [d for d in os.listdir(dataset_path) 
                   if os.path.isdir(os.path.join(dataset_path, d)) and len(d) == 15]
        if not exp_dirs:
            continue
            
        latest_exp = sorted(exp_dirs, reverse=True)[0]
        scalars_file = os.path.join(dataset_path, latest_exp, 'vis_data', 'scalars.json')
        
        if not os.path.exists(scalars_file):
            continue
            
        # 解析scalars.json
        with open(scalars_file) as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'coco/bbox_mAP' in data:
                        results[dataset].append(round(data['coco/bbox_mAP'], 3))
                except json.JSONDecodeError:
                    continue
                    
    """保存为要求的JSON格式"""
    with open(output_file, 'w') as f:
        for dataset in sorted(results.keys(), key=lambda x: x.lower()):
            json.dump(
                {dataset: results[dataset]},
                f,
                ensure_ascii=False,
                separators=(',', ': ')
            )
            f.write('\n')  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default="work_dirs/CDIOD/minival_adapter_1e-3d128s1")
    parser.add_argument('--output_file', default="CDIOD_results_per_epoch.json")
    args = parser.parse_args()
    output_file = os.path.join(args.root_dir, args.output_file)
    # 执行收集和保存
    aggregate_results_per_epoch(
        root_dir=args.root_dir,
        output_file=output_file
    )
