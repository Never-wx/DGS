import os

import torch
import torch.distributed as dist
import yaml
from mmengine.logging import MMLogger

from .adaptive_domain_predictor import AdaptiveDomainPredictor_Dist, gather_img_emebedding

# ---------------------------------------------------------------------------
# Utility function
# ---------------------------------------------------------------------------

def _svd_regularize_cov(cov: torch.Tensor, min_eig_ratio: float = 1e-3) -> torch.Tensor:
    """
    Regularize a covariance matrix via eigendecomposition truncation.

    Clamps all eigenvalues below min_eig_ratio * max_eigenvalue to prevent
    numerical singularity when the matrix is rank-deficient (N < D).

    Args:
        cov: [D, D] symmetric covariance matrix (possibly rank-deficient).
        min_eig_ratio: Floor ratio relative to the largest eigenvalue.
                       Recommended range [1e-4, 1e-2]. Default 1e-3.

    Returns:
        reg_cov: [D, D] positive-definite regularized covariance matrix.

    Notes:
        - Symmetrises the input first to eliminate floating-point asymmetry.
        - Uses torch.linalg.eigh which is optimised for symmetric matrices.
        - Unlike Ledoit-Wolf, the shrinkage target is the covariance itself
          (self-referenced), so inter-task covariance differences are preserved.
    """
    # Symmetrise to remove floating-point asymmetry from torch.cov
    cov = (cov + cov.T) / 2.0

    # eigh is more numerically stable than eig for symmetric matrices
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # Floor: clamp eigenvalues to min_eig_ratio * max_eigenvalue
    max_eig = eigenvalues.max()
    # max_eig = max_eig.clamp(min=1e-10)  # guard against all-zero or near-zero matrices
    min_eig = max_eig * min_eig_ratio
    eigenvalues_reg = torch.clamp(eigenvalues, min=min_eig.item())

    # Reconstruct: V @ diag(lambda_reg) @ V^T
    reg_cov = eigenvectors @ torch.diag(eigenvalues_reg) @ eigenvectors.T
    return reg_cov


class AdaptiveDomainPredictor_SVD(AdaptiveDomainPredictor_Dist):
    """Domain predictor using SVD-regularized covariance and symmetric KL routing."""

    def __init__(self, domain_predictor_cfg, num_tasks,
                 moe_modules=None, task_id=None, seen_tasks=None, work_dirs=None):
        # Read min_eig_ratio before super().__init__ because super() calls
        # load_task_stats -> estimate_task_stats which needs this attribute.
        self.min_eig_ratio = getattr(domain_predictor_cfg, 'min_eig_ratio', 1e-3)
        super().__init__(
            domain_predictor_cfg=domain_predictor_cfg,
            num_tasks=num_tasks,
            moe_modules=moe_modules,
            task_id=task_id,
            seen_tasks=seen_tasks,
            work_dirs=work_dirs,
        )

    def estimate_task_stats(self, feat_path, stats_path, task_name):
        """Estimate and save SVD-regularized covariance statistics for a task."""
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        feat_path_full = os.path.join(feat_path, task_name)
        # Fallback to train/ subdirectory if no .pt files in flat directory
        if not any(f.endswith('.pt') for f in os.listdir(feat_path_full)):
            feat_path_full = os.path.join(feat_path_full, 'train')
        features = gather_img_emebedding(feat_path_full)

        num_samples = len(features)

        if features.dim() > 2:
            # Multilevel features: [num_samples, num_levels, feat_dim]
            num_levels = features.size(1)
            multilevel_mean = []
            multilevel_cov = []
            multilevel_template_cov = []
            multilevel_kl_div = []

            for level in range(num_levels):
                level_features = features[:, level, :]  # [N, D]
                level_mean = torch.mean(level_features, dim=0)

                raw_cov = torch.cov(level_features.T) + self.eps * torch.eye(
                    level_features.size(-1), device=level_features.device
                )
                level_cov = _svd_regularize_cov(raw_cov, self.min_eig_ratio)

                level_template_cov = torch.eye(level_mean.shape[0]) * level_cov.max().item() / 30

                dist1 = torch.distributions.MultivariateNormal(level_mean, level_cov)
                rand_ix = torch.randperm(len(level_features))[:20]
                dist2 = torch.distributions.MultivariateNormal(
                    level_features[rand_ix].mean(dim=0), level_cov
                )
                level_kl_div = (
                    torch.distributions.kl.kl_divergence(dist1, dist2)
                    + torch.distributions.kl.kl_divergence(dist2, dist1)
                ) / 2

                multilevel_mean.append(level_mean)
                multilevel_cov.append(level_cov)
                multilevel_template_cov.append(level_template_cov)
                multilevel_kl_div.append(level_kl_div)

            stats = {
                'mean': torch.stack(multilevel_mean, dim=0),
                'cov': torch.stack(multilevel_cov, dim=0),
                'template_cov': torch.stack(multilevel_template_cov, dim=0),
                'kl_div': torch.stack(multilevel_kl_div, dim=0),
                'num_samples': num_samples,
            }

        else:
            # Scalar (single-level) features: [num_samples, feat_dim]
            task_mean = torch.mean(features, dim=0)

            raw_cov = torch.cov(features.T) + self.eps * torch.eye(
                features.size(-1), device=features.device
            )
            task_cov = _svd_regularize_cov(raw_cov, self.min_eig_ratio)

            template_cov = torch.eye(task_mean.shape[0]) * task_cov.max().item() / 30

            dist1 = torch.distributions.MultivariateNormal(task_mean, task_cov)
            rand_ix = torch.randperm(len(features))[:20]
            dist2 = torch.distributions.MultivariateNormal(
                features[rand_ix].mean(dim=0), task_cov
            )
            kl_div = (
                torch.distributions.kl.kl_divergence(dist1, dist2)
                + torch.distributions.kl.kl_divergence(dist2, dist1)
            ) / 2

            stats = {
                'mean': task_mean,
                'cov': task_cov,
                'template_cov': template_cov,
                'kl_div': kl_div,
                'num_samples': num_samples,
            }

        del features
        torch.save(stats, stats_path)
        return stats

    def taskwise_adaptive_expansion(self, moe_modules):
        """Group tasks by symmetric KL distance between task distributions."""
        cur_task_id = len(self.seen_tasks) - 1
        logger: MMLogger = MMLogger.get_current_instance()

        if cur_task_id not in self.task_id_mapping.keys():
            _, _, cur_task_dist = self.get_task_distribution(cur_task_id)

            distances = []
            for t_id in range(cur_task_id):
                _, task_cov, task_dist = self.get_task_distribution(t_id)

                if self.multilevel:
                    level_kl = []
                    for cur_lvl, tsk_lvl in zip(cur_task_dist, task_dist):
                        kl_lvl = (
                            torch.distributions.kl.kl_divergence(cur_lvl, tsk_lvl)
                            + torch.distributions.kl.kl_divergence(tsk_lvl, cur_lvl)
                        ) / 2
                        level_kl.append(kl_lvl)
                    d = torch.mean(torch.stack(level_kl))
                else:
                    d = (
                        torch.distributions.kl.kl_divergence(cur_task_dist, task_dist)
                        + torch.distributions.kl.kl_divergence(task_dist, cur_task_dist)
                    ) / 2
                distances.append(d)

            distances_tensor = torch.stack(distances)
            logger.info(f'distances with all seen tasks: {distances_tensor}')
            min_dist, closest_task_id = torch.min(distances_tensor, dim=0)

            if min_dist < self.expand_th:
                task_group_id = self.task_id_mapping[closest_task_id.item()]
                self.task_id_mapping[cur_task_id] = task_group_id
                logger.info(
                    f'Task {cur_task_id} will be merged to previous task '
                    f'{self.task_id_mapping[closest_task_id.item()]}'
                )
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

        # Generate group->task mapping
        group2tasks = {}
        for k, v in self.task_id_mapping.items():
            if v not in group2tasks:
                group2tasks[v] = []
            group2tasks[v].append(k)
        self.group2tasks = group2tasks

        for module in moe_modules:
            if hasattr(module, 'group_init'):
                module.group_init(
                    task2group=self.task_id_mapping,
                    group2tasks=self.group2tasks,
                )

    @torch.no_grad()
    def forward(self, features, task_id=None, input_sample=None):
        if self.training:
            return None

        if self.multilevel:
            num_levels = len(features)
            features_stacked = torch.stack(
                [torch.nn.functional.adaptive_avg_pool2d(f, 1).squeeze() for f in features]
            )  # [num_levels, D]
            cur_mean = features_stacked.mean(dim=1) if features_stacked.dim() > 2 else features_stacked

            kl_divs = []
            for t_id in range(len(self.seen_tasks)):
                task_mean, task_cov, task_dist = self.get_task_distribution(t_id)
                level_kl = []
                for level in range(num_levels):
                    level_cur_dist = torch.distributions.MultivariateNormal(
                        cur_mean[level], task_cov[level]
                    )
                    kl = (
                        torch.distributions.kl.kl_divergence(level_cur_dist, task_dist[level])
                        + torch.distributions.kl.kl_divergence(task_dist[level], level_cur_dist)
                    ) / 2
                    level_kl.append(kl)
                kl_divs.append(torch.mean(torch.stack(level_kl)))

        else:
            features_last = torch.nn.functional.adaptive_avg_pool2d(features[-1], 1).squeeze()
            cur_mean = features_last.mean(dim=0) if features_last.dim() > 1 else features_last

            kl_divs = []
            for t_id in range(len(self.seen_tasks)):
                task_mean, task_cov, task_dist = self.get_task_distribution(t_id)
                cur_dist = torch.distributions.MultivariateNormal(cur_mean, task_cov)
                kl = (
                    torch.distributions.kl.kl_divergence(cur_dist, task_dist)
                    + torch.distributions.kl.kl_divergence(task_dist, cur_dist)
                ) / 2
                kl_divs.append(kl)

        kl_divs_tensor = torch.stack(kl_divs)
        min_kl, pred_task_id = torch.min(kl_divs_tensor, dim=0)

        pred_task_id = pred_task_id if min_kl < self.ood_th else -1

        self._record_acc(input_sample, pred_task_id, pred_value=kl_divs_tensor)
        return int(pred_task_id)



