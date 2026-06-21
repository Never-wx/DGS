import torch.nn.functional
import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import json
import yaml
import torch.distributed as dist
from mmengine.dist import master_only, is_main_process
from mmengine.logging import MMLogger


def gather_img_emebedding(feat_path, max_sample=None):
    """
    Gather image embeddings from .pt files in the given directory.
    
    Args:
        feat_path (str): Path to directory containing .pt feature files
        max_sample (int, optional): Maximum number of samples to load
        
    Returns:
        torch.Tensor: Stacked features from all files
    """
    # Get all .pt files in directory
    imgfeat_files = sorted([f for f in os.listdir(feat_path) if f.endswith('.pt')])
    
    # Load all features into a list
    feat_list = []
    for filename in imgfeat_files:
        filepath = os.path.join(feat_path, filename)
        # Load directly as torch tensor
        feat = torch.load(filepath)
        feat_list.append(feat)
    
    # Stack all features into single tensor
    features = torch.stack(feat_list, dim=0)
    
    # Apply max_sample if specified
    if max_sample is not None and len(features) > max_sample:
        rand_idx = torch.randperm(len(features))[:max_sample]
        features = features[rand_idx]
    
    return features

class AdaptiveDomainPredictor_Dist(nn.Module):
    """KL-based domain predictor for dynamic task grouping and routing."""

    def __init__(self, domain_predictor_cfg, num_tasks,
                 moe_modules=None, 
                 task_id=None, 
                 seen_tasks=None,
                 work_dirs=None):
        super().__init__()
        self.domain_predictor_cfg = domain_predictor_cfg
        self.num_tasks = num_tasks
        self.task_id = task_id
        self.seen_tasks = seen_tasks
        self.work_dirs=work_dirs
        self.multilevel = domain_predictor_cfg.multilevel
        self.expand_th = domain_predictor_cfg.expand_th
        self.ood_th = domain_predictor_cfg.ood_th   # KL divergence threshold
        self.eps = 1e-6  # Initialize with small identity matrix for numerical stability
        self.task_id_mapping = self.init_task_id_mapping()
        self.tasks = self._cls2task()
        self.load_task_stats()
        self.taskwise_adaptive_expansion(moe_modules)
        self.register_buffer('accurate_pred', torch.zeros([self.num_tasks + 1], dtype=torch.int64), persistent=False)
        self.register_buffer('num_samples', torch.zeros([self.num_tasks + 1], dtype=torch.int64), persistent=False)
        self.register_buffer('distance', torch.zeros([self.num_tasks + 1]), persistent=False)
        self.register_buffer('max_distance', torch.zeros([self.num_tasks + 1]), persistent=False)
        self.register_buffer('min_distance', torch.ones([self.num_tasks + 1]) * 1000, persistent=False)
    
    def _cls2task(self):
        """Map dataset labels or names to task ids for routing diagnostics."""
        if self.num_tasks == 5:
            tasks = {**{i: 0 for i in range(0, 10)},
                    **{i: 1 for i in range(10, 20)},
                    **{i: 2 for i in range(20, 30)},
                    **{i: 3 for i in range(30, 40)},
                    **{i: 4 for i in range(40, 45)},
                    **{i: 5 for i in range(45, 50)}}
            
        elif self.num_tasks == 10:
            tasks = {**{i: 0 for i in range(0, 5)},
                    **{i: 1 for i in range(5, 10)},
                    **{i: 2 for i in range(10, 15)},
                    **{i: 3 for i in range(15, 20)},
                    **{i: 4 for i in range(20, 25)},
                    **{i: 5 for i in range(25, 30)},
                    **{i: 6 for i in range(30, 35)},
                    **{i: 7 for i in range(35, 40)},
                    **{i: 8 for i in range(40, 45)},
                    **{i: 9 for i in range(45, 50)}}
            
        elif self.num_tasks == 13:
            tasks = {'AerialMaritimeDrone':0, 'Aquarium':1, 'CottontailRabbits':2, 'EgoHands':3,
                     'NorthAmericaMushroom':4, 'Packages':5, 'PascalVOC':6, 'pistols':7, 'pothole':8, 'Raccoon':9,
                     'ShellfishOpenImages':10, 'thermalDogsAndPeople':11, 'VehiclesOpenImages':12} 

        else:
            tasks = None    

        return tasks
    
    def _record_acc(self, input_sample, pred_task_id, pred_value, group_wise=False): 
        """ Task_ids prediction Analysis """
        if input_sample is not None and self.tasks is not None:
            for sample in input_sample:
                dataset_name = sample.img_path.split('/')[2]
                if sample.img_path.split('/')[1] == 'CDIOD' and len(sample.gt_instances.labels) > 0:
                    labels = sample.gt_instances.labels 
                    dataset_name = torch.argmax(labels.bincount()).item()
                
                if group_wise:
                    gt_task_id = self.tasks.get(dataset_name, -1)
                    gt_group_id = self.task_id_mapping.get(gt_task_id ,-1)
                    pred_group_id = self.task_id_mapping.get(int(pred_task_id) ,-1)

                    self.num_samples[gt_group_id] += 1
                    self.accurate_pred[gt_group_id] += (gt_group_id == pred_group_id)
                    distance = pred_value[gt_group_id] if gt_group_id > -1 else torch.min(pred_value) 
                    self.distance[gt_group_id] += distance
                    self.max_distance[gt_group_id] = max(self.max_distance[gt_group_id], distance)
                    self.min_distance[gt_group_id] = min(self.min_distance[gt_group_id], distance)                
                
                else:
                    gt_task_id = self.tasks.get(dataset_name, -1)
                    if gt_task_id <= self.task_id:
                        self.num_samples[gt_task_id] += 1
                        self.accurate_pred[gt_task_id] += (gt_task_id == pred_task_id)
                        distance = pred_value[gt_task_id] if gt_task_id > -1 else torch.min(pred_value) 
                        self.distance[gt_task_id] += distance
                        self.max_distance[gt_task_id] = max(self.max_distance[gt_task_id], distance)
                        self.min_distance[gt_task_id] = min(self.min_distance[gt_task_id], distance)
    
    def estimate_task_stats(self, feat_path, stats_path, task_name):
        """estimate and save statistics for seen tasks"""
        if not os.path.exists(os.path.dirname(stats_path)):
            os.makedirs(os.path.dirname(stats_path), exist_ok=True)     
        feat_path = os.path.join(feat_path, task_name)
        # Fallback to train/ subdirectory if no .pt files in flat directory
        if not any(f.endswith('.pt') for f in os.listdir(feat_path)):
            feat_path = os.path.join(feat_path, 'train')
        # Calculate mean and covariance for current task features
        features = gather_img_emebedding(feat_path)  
        if features.dim() > 2:  # multilevel features, shape: [num_samples, num_levels, feat_dim]
            stats = {}
            num_levels = features.size(1)
            multilevel_mean = []
            multilevel_cov = []
            multilevel_template_cov = []
            multilevel_kl_div = []
            for level in range(num_levels):
                level_features = features[:, level, :]  # [num_samples, feat_dim]

                # Calculate statistics for this level
                level_mean = torch.mean(level_features, dim=0)
                level_cov = torch.cov(level_features.T) + self.eps * torch.eye(level_features.size(-1), device=level_features.device)
                level_template_cov = torch.eye(level_mean.shape[0]) * level_cov.max().item() / 30
                # Calculate KL divergence
                dist1 = torch.distributions.MultivariateNormal(level_mean, level_cov)
                rand_ix = torch.randperm(len(level_features))[:20]
                dist2 = torch.distributions.MultivariateNormal(level_features[rand_ix].mean(dim=0), level_cov)
                level_kl_div = (torch.distributions.kl.kl_divergence(dist1, dist2) + torch.distributions.kl.kl_divergence(dist2, dist1)) / 2  
                multilevel_mean.append(level_mean)
                multilevel_cov.append(level_cov)
                multilevel_template_cov.append(level_template_cov)
                multilevel_kl_div.append(level_kl_div)

            stats = {
                'mean': torch.stack(multilevel_mean, dim=0),
                'cov': torch.stack(multilevel_cov, dim=0),
                'template_cov': torch.stack(multilevel_template_cov, dim=0),
                'kl_div': torch.stack(multilevel_kl_div, dim=0)
            }
        else:
            # if features.dim() > 2:
            #     features = features[:, -1, :]   # extract last layer feature

            task_mean = torch.mean(features, dim=0)
            task_cov = torch.cov(features.T) + self.eps * torch.eye(features.size(-1), device=features.device)
            template_cov = torch.eye(task_mean.shape[0]) * task_cov.max().item() / 30
            # in-domian kl_div
            dist1 = torch.distributions.MultivariateNormal(task_mean, task_cov)
            rand_ix = torch.randperm(len(features))[:20]
            dist2 = torch.distributions.MultivariateNormal(features[rand_ix].mean(dim=0), task_cov)
            kl_div = (torch.distributions.kl.kl_divergence(dist1, dist2) + torch.distributions.kl.kl_divergence(dist2, dist1)) / 2
            
            # Save stats
            stats = {
                'mean': task_mean,
                'cov': task_cov,
                'template_cov': template_cov,
                'kl_div': kl_div,
            }
        del features
        torch.save(stats, stats_path)
        return stats
    
    def load_task_stats(self):
        """Load pre-computed statistics for each task"""
        logger: MMLogger = MMLogger.get_current_instance()
        
        for task_id in range(len(self.seen_tasks)): 
            task_name = self.seen_tasks[task_id]
            stats_file = os.path.join(self.domain_predictor_cfg.stats_path, f'{task_name}.pt')
            if os.path.exists(stats_file):
                logger.info(f'Loading task {task_name} stats from {stats_file}')
                stats = torch.load(stats_file)
            else:
                logger.info(f'Not found task {task_name} stats, estimating from {self.domain_predictor_cfg.feat_path}')
                stats = self.estimate_task_stats(feat_path=self.domain_predictor_cfg.feat_path, 
                                                 stats_path=stats_file, 
                                                 task_name=task_name)

            # Register as buffers to ensure proper device placement
            self.register_buffer(f'mean_{task_id}', stats['mean'], persistent=False)
            self.register_buffer(f'cov_{task_id}', stats['cov'], persistent=False)
            # self.register_buffer(f'kl_div_{task_id}', stats['kl_div'], persistent=False)
    
    def get_task_distribution(self, task_id):
        """Get multivariate normal distribution for a task"""
        if self.multilevel: 
            mean = getattr(self, f'mean_{task_id}')
            cov = getattr(self, f'cov_{task_id}')
            dist = [torch.distributions.MultivariateNormal(mean[level], cov[level]) for level in range(mean.size(0))]
        else:
            mean = getattr(self, f'mean_{task_id}')[-1]
            cov = getattr(self, f'cov_{task_id}')[-1] 
            dist = torch.distributions.MultivariateNormal(mean, cov)
        return mean, cov, dist
    
    def init_task_id_mapping(self):
        """Load the task-to-group mapping used by grouped experts."""
        cur_task_id = len(self.seen_tasks) - 1
        logger: MMLogger = MMLogger.get_current_instance()

        if hasattr(self.domain_predictor_cfg, 'task_id_mapping_path'): 
            task_mapping_file = self.domain_predictor_cfg.task_id_mapping_path
        else:
            work_dir = self.work_dirs if self.work_dirs is not None \
                else os.path.dirname(os.path.dirname(os.path.dirname(logger.log_file)))
            task_mapping_file = os.path.join(work_dir, 'task_id_mapping.yaml')   
        
        self.task_id_mapping_path = task_mapping_file

        if cur_task_id == 0:   
            task_id_mapping = {cur_task_id: cur_task_id} 

        elif os.path.isfile(task_mapping_file):
            with open(task_mapping_file, "r") as f: 
                loaded_mapping = yaml.safe_load(f)

            if loaded_mapping is None:
                task_id_mapping = {}
                logger.info(f'empty task_id_mapping file, path is {task_mapping_file}')
            else:
                task_id_mapping = loaded_mapping
                # Filter out task IDs beyond current task
                filtered_mapping = {k: v for k, v in task_id_mapping.items() if int(k) <= cur_task_id}
                if filtered_mapping != task_id_mapping:
                    logger.info(f'Filtered task_id_mapping to include only seen tasks (max={cur_task_id})')
                    task_id_mapping = filtered_mapping
                
                for k, v in task_id_mapping.items():
                    logger.info(f'task_id_mapping file {k}:{v}')
        else:
            raise FileNotFoundError(f"Task ID mapping file not found: {task_mapping_file}")
        
        return task_id_mapping
    
    def taskwise_adaptive_expansion(self, moe_modules):
        """Merge the current task into a close group or allocate a new group."""
        cur_task_id = len(self.seen_tasks) - 1
        logger: MMLogger = MMLogger.get_current_instance()
        if cur_task_id not in self.task_id_mapping.keys():
            kl_divs = []
            _, _, cur_task_dist = self.get_task_distribution(cur_task_id)
            for t_id in range(cur_task_id):
                _, _, task_dist = self.get_task_distribution(t_id)

                if self.multilevel:
                    level_kl = []
                    for cur_task_dist_level, task_dist_level in zip(cur_task_dist, task_dist):
                        kl_level = (torch.distributions.kl.kl_divergence(cur_task_dist_level, task_dist_level) + \
                                torch.distributions.kl.kl_divergence(task_dist_level, cur_task_dist_level)) / 2                         
                        level_kl.append(kl_level)
                    
                    kl = torch.mean(torch.stack(level_kl), dim=0)

                else:
                    kl = (torch.distributions.kl.kl_divergence(cur_task_dist, task_dist) + \
                            torch.distributions.kl.kl_divergence(task_dist, cur_task_dist)) / 2 
                
                kl_divs.append(kl)
            
            kl_divs = torch.stack(kl_divs)  # kl_divs between cur_task and all seen tasks
            logger.info(f'kl_divs with all seen tasks: {kl_divs}')
            min_kl, closest_task_id = torch.min(kl_divs, dim=0)
            if min_kl < self.expand_th:
                task_group_id = self.task_id_mapping[closest_task_id.item()]
                self.task_id_mapping[cur_task_id] = task_group_id  # map cur_task_id to previous group
                logger.info(f'Task {cur_task_id} will be merged to previous task {self.task_id_mapping[closest_task_id.item()]}')
            else:  
                expand_group_id = max(self.task_id_mapping.values()) + 1              
                self.task_id_mapping[cur_task_id] = expand_group_id
                logger.info(f'Task {cur_task_id} will be expanded to new group {expand_group_id}')
        
        if dist.is_available() and dist.is_initialized():
            dist.barrier()  
        
        try:
            with open(self.task_id_mapping_path, "w") as f:
                yaml.dump(self.task_id_mapping, f)
            logger.info(f"Successfully updated task_id_mapping to {self.task_id_mapping}")
        except IOError as e:
            logger.error(f"Failed to write task_id_mapping file: {e}")
        

        # generate group->task->expert mapping
        group2tasks = {}  # {group_id: [task_ids]}
        for k, v in self.task_id_mapping.items():
            if v not in group2tasks:
                group2tasks[v] = []
            group2tasks[v].append(k)    
        self.group2tasks = group2tasks 
        
        # Re-initialize grouped experts after updating task-to-group assignment.
        for module in moe_modules:
            if hasattr(module, 'group_init'):
                module.group_init(task2group=self.task_id_mapping,
                                  group2tasks=self.group2tasks)  
    
    @torch.no_grad()
    def forward(self, features, task_id=None, input_sample=None):
        if self.training:
            return None
        else:  # Test mode: compute KL divergence with all task distributions
            if self.multilevel:
                num_levels = len(features)
                features = torch.stack([torch.nn.functional.adaptive_avg_pool2d(feat_lvl, 1).squeeze() for feat_lvl in features])  
                cur_mean = torch.mean(features, dim=1) if features.dim() > 2 else features  # [lvl, D]

                kl_divs = []
                for t_id in range(len(self.seen_tasks)):
                    task_mean, task_cov, task_dist = self.get_task_distribution(t_id)
                    level_kl = []
                    for level in range(num_levels):
                        level_task_dist = task_dist[level]
                        level_cur_dist = torch.distributions.MultivariateNormal(cur_mean[level], task_cov[level])
                        kl = (torch.distributions.kl.kl_divergence(level_cur_dist, level_task_dist) + \
                              torch.distributions.kl.kl_divergence(level_task_dist, level_cur_dist)) / 2
                        level_kl.append(kl)
                    level_kl_mean = torch.mean(torch.stack(level_kl), dim=0)
                    kl_divs.append(level_kl_mean)
            else:             
                features = torch.nn.functional.adaptive_avg_pool2d(features[-1], 1).squeeze()  # [B, D]
                cur_mean = features.mean(dim=0) if features.dim() > 1 else features
                kl_divs = []
                # Compare with each task's distribution
                for t_id in range(len(self.seen_tasks)):
                    task_mean, task_cov, task_dist = self.get_task_distribution(t_id)
                    cur_dist = torch.distributions.MultivariateNormal(cur_mean, task_cov)   
                    kl = (torch.distributions.kl.kl_divergence(cur_dist, task_dist) + \
                          torch.distributions.kl.kl_divergence(task_dist, cur_dist)) / 2
                    kl_divs.append(kl)
            
            kl_divs = torch.stack(kl_divs)
            min_kl, pred_task_id = torch.min(kl_divs, dim=0)
            
            # Apply threshold - if divergence is too high, mark as unknown (-1)
            pred_task_id = pred_task_id if min_kl < self.ood_th else -1
            
            self._record_acc(input_sample, pred_task_id, pred_value=kl_divs)
            return int(pred_task_id)
