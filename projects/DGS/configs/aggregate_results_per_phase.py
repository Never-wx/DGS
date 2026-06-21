import os
import json
from collections import defaultdict
import argparse

def collect_additional_metrics(root_dir):
    """收集ZCOCO和ZODINW13的评估结果（深度搜索）"""
    def find_json(task_path):
        if not os.path.isdir(task_path):
            return None
            
        # 查找所有时间戳格式的目录（如20250220_111930）
        time_dirs = [d for d in os.listdir(task_path) 
                    if os.path.isdir(os.path.join(task_path, d)) and len(d) == 15]
        if not time_dirs:
            return None
            
        latest_time_dir = sorted(time_dirs, reverse=True)[0]
        file_path = os.path.join(task_path, latest_time_dir)
        
        if not os.path.exists(file_path):
            return None
            
        json_files = [f for f in os.listdir(file_path) 
                     if f.endswith('.json') and os.path.isfile(os.path.join(file_path, f))]
        if not json_files:
            return None
            
        # 取第一个找到的json文件（假设只有一个）
        return os.path.join(file_path, json_files[0])

    metrics = {'ZCOCO': [], 'ZODINW13_avg': []}

    # 处理ZCOCO
    zcoco_dir = os.path.join(root_dir, "ZCOCO")
    result_file = find_json(zcoco_dir)
    if result_file:
        try:
            with open(result_file) as f:
                data = json.load(f)
                metrics['ZCOCO'].append(round(data.get('coco/bbox_mAP', 0), 3))
        except Exception as e:
            print(f"[ZCOCO] 结果文件解析失败: {str(e)}")

    # 处理ZODINW13
    zodinw_dir = os.path.join(root_dir, "ZODinW13")
    result_file = find_json(zodinw_dir)
    if result_file:
        try:
            with open(result_file) as f:
                data = json.load(f)
                metrics['ZODINW13_avg'].append(round(data.get('avg/coco/bbox_mAP', 0), 3))
        except Exception as e:
            print(f"[ZODINW13] 结果文件解析失败: {str(e)}")

    return metrics

def aggregate_results(phases, root_dir, output_file):
    """
    参数:
    phases - 训练步骤对应的目录名列表 (如 ['animals', 'cityscape'])
    root_dir - 所有训练步骤目录的根路径
    output_file - 结果输出路径
    """
    # 初始化结果字典（保持数据集顺序）
    prefix = [phase.split('_Task')[0] for phase in phases]
    valid_datasets = list(dict.fromkeys(prefix))
    final_results = {dataset: [] for dataset in valid_datasets}
    final_results["avg"] = []  # 添加avg字段
    final_results.update({task: [] for task in phases})

    for step_dir in phases:
        step_path = os.path.join(root_dir, step_dir)
        
        # 获取实验目录列表并找到最新的
        experiment_dirs = [d for d in os.listdir(step_path) 
                          if os.path.isdir(os.path.join(step_path, d))]
        experiment_dirs.sort()
        latest_experiment = experiment_dirs[-1]

        # 构建scalars.json路径
        scalars_path = os.path.join(step_path, latest_experiment, 'vis_data', 'scalars.json')
        
        # 读取最后一行数据
        with open(scalars_path, 'r') as f:
            last_line = ''
            for line in f:
                if line.strip():
                    last_line = line.strip()
            
            data = json.loads(last_line)
            
            # 收集当前步骤所有指标
            step_metrics = {}
            for metric_key in data:
                if metric_key.endswith('/bbox_mAP'):
                    key_list = metric_key.split('/')
                    dataset = key_list[0] if len(key_list) < 4 else key_list[1]
                    step_metrics[dataset] = round(data[metric_key], 3)
            
            # 填充数据集结果并计算avg
            total = 0
            valid_count = 0
            
            for dataset in valid_datasets:
                value = step_metrics.get(dataset, 0.0)
                final_results[dataset].append(value)
                if value > 0:  # 仅统计有效值
                    total += value
                    valid_count += 1

            # per task performance
            for task in phases:
                value = step_metrics.get(task, 0.0)
                final_results[task].append(value)

            # 计算并保存平均
            avg = round(total / valid_count, 3) if valid_count > 0 else 0.0
            final_results["avg"].append(avg)
    
    # 收集额外指标（直接使用当前root_dir）
    additional_metrics = collect_additional_metrics(root_dir)
    
    # 重组结果结构
    per_dataset = {dataset: final_results[dataset] for dataset in valid_datasets}
    per_dataset["avg"] = final_results["avg"]  # 单独添加avg到阶段性能
    per_task = {task: final_results[task] for task in phases}

    # 新增最佳性能计算
    best_performance = {}
    total_best = 0.0
    valid_count = 0
    
    for dataset, values in per_dataset.items():
        if dataset != "avg":  # 排除平均值数据集
            best_value = round(max(values), 3)
            best_performance[dataset] = best_value
            total_best += best_value
            valid_count += 1
    
    # 计算平均最佳性能（排除avg自身）
    if valid_count > 0:
        best_performance["avg"] = round(total_best / valid_count, 3)
    else:
        best_performance["avg"] = 0.0

    # 重构final_performance结构
    last_performance = {
        dataset: values[-1] 
        for dataset, values in per_dataset.items()
        if isinstance(values, list)
    }
    
    results = {
        "per_phase_performance_datasetwise": per_dataset,
        "last_performance": last_performance,
        "best_performance": best_performance,
        "per_phase_performance_taskwise": per_task,
        # "zero_shot_performance": {k: v for k, v in additional_metrics.items() if v}
    }
    
    # 写入文件
    with open(output_file, 'w') as f:
        json_str = json.dumps(
            results,
            indent=2,
            separators=(',', ': '),
            ensure_ascii=False
        )
        f.write(json_str)
        f.write('\n')  # 添加结尾换行符

def aggregate_results_lines(phases, root_dir, output_file):
    """
    参数:
    phases - 训练步骤对应的目录名列表
    root_dir - 所有训练步骤目录的根路径
    output_file - 结果输出路径
    """
    # 初始化结果字典（保持数据集顺序）
    final_results = {dataset: [] for dataset in phases}
    final_results["avg"] = []  # 添加avg字段
    
    for step_dir in phases:
        step_path = os.path.join(root_dir, step_dir)
        
        # 获取实验目录列表并找到最新的
        experiment_dirs = [d for d in os.listdir(step_path) 
                          if os.path.isdir(os.path.join(step_path, d))]
        experiment_dirs.sort()
        latest_experiment = experiment_dirs[-1]

        # 构建scalars.json路径
        scalars_path = os.path.join(step_path, latest_experiment, 'vis_data', 'scalars.json')
        
        # 读取最后一行数据
        with open(scalars_path, 'r') as f:
            last_line = ''
            for line in f:
                if line.strip():
                    last_line = line.strip()
            
            data = json.loads(last_line)
            
            # 收集当前步骤所有指标
            step_metrics = {}
            for metric_key in data:
                if metric_key.endswith('/bbox_mAP'):
                    dataset = metric_key.split('/')[0]
                    step_metrics[dataset] = round(data[metric_key], 3)
            
            # 填充数据集结果并计算avg
            total = 0
            valid_count = 0
            for dataset in phases:
                value = step_metrics.get(dataset, 0.0)
                final_results[dataset].append(value)
                if value > 0:  # 仅统计有效值
                    total += value
                    valid_count += 1
            
            # 计算并保存平均
            avg = round(total / valid_count, 3) if valid_count > 0 else 0.0
            final_results["avg"].append(avg)

    # 收集额外指标（直接使用当前root_dir）
    additional_metrics = collect_additional_metrics(root_dir)
    
    # 保存结果（每行一个数据集/avg）
    with open(output_file, 'w') as f:
        # Performance of each tasks after per incremental step
        for dataset in phases:
            json.dump({dataset: final_results[dataset]}, f, ensure_ascii=False)
            f.write('\n')
        
        json.dump({"avg": final_results["avg"]}, f, ensure_ascii=False)
        f.write('\n')
        
        # Performance of each tasks after all incremental phases
        last_values = [final_results[dataset][-1] for dataset in phases]
        last_values.append(final_results["avg"][-1])
        json.dump({"Last_performance": last_values}, f, ensure_ascii=False)
        f.write('\n')
        
        # zero-shot performance
        if additional_metrics['ZCOCO']:
            json.dump({"ZCOCO": additional_metrics['ZCOCO']}, f, ensure_ascii=False)
            f.write('\n')
        if additional_metrics['ZODINW13_avg']:
            json.dump({"ZODINW13_avg": additional_metrics['ZODINW13_avg']}, f, ensure_ascii=False)
            f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--phases', default=['animals', 'cityscape', 'clipart1k', 'construction_safety',
    #                                         'DIOR', 'document_parts', 'liver_disease', 'NEU-DET', 'RUOD', 
    #                                         'washroom'], type=eval)
    parser.add_argument('--phases', default=['DIOR_Task2-1', 'DIOR_Task2-2', 'PascalVOC_Task2-1', 'PascalVOC_Task2-2', 'RUOD_Task2-1', 'RUOD_Task2-2'], type=eval)
    parser.add_argument('--root_dir', default="work_dirs/CDIOD/moe/moe_mergeall_lora_ffn_exp1k1r8_learned")
    parser.add_argument('--output_file', default="CDIOD_results_per_phase.json")
    args = parser.parse_args()
    output_file = os.path.join(args.root_dir, args.output_file)
    
    aggregate_results(
        phases=args.phases,
        root_dir=args.root_dir,
        output_file=output_file
    )


