# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict, defaultdict
from typing import Sequence, Union

from mmengine.dist import (broadcast_object_list, collect_results,
                           is_main_process)
from mmengine.evaluator import BaseMetric, Evaluator
from mmengine.evaluator.metric import _to_cpu
from mmengine.registry import EVALUATOR
from mmengine.logging import MMLogger
from mmdet.utils import ConfigType

@EVALUATOR.register_module()
class CDIODMultiDatasetsEvaluator(Evaluator):
    """Wrapper class to compose class: `ConcatDataset` and multiple
    :class:`BaseMetric` instances.
    The metrics will be evaluated on each dataset slice separately. The name of
    the each metric is the concatenation of the dataset prefix, the metric
    prefix and the key of metric - e.g.
    `dataset_prefix/metric_prefix/accuracy`.

    Args:
        metrics (dict or BaseMetric or Sequence): The config of metrics.
        dataset_prefixes (Sequence[str]): The prefix of each dataset. The
            length of this sequence should be the same as the length of the
            datasets.
    """

    def __init__(self, metrics: Union[ConfigType, BaseMetric, Sequence],
                 dataset_prefixes: Sequence[str],
                 evaluation_tasks: str = None) -> None:
        super().__init__(metrics)
        self.dataset_prefixes = dataset_prefixes
        self._setups = False
        self.evaluation_tasks = set(self.str2list(evaluation_tasks)) if evaluation_tasks is not None else None
    
    def str2list(self, datasets):
        if type(datasets) is not tuple:
            if type(datasets) is str:
                # Split space-separated strings into a list
                datasets = datasets.split(',')
            datasets = tuple(datasets)
        return datasets

    def _get_cumulative_sizes(self):
        # ConcatDataset have a property `cumulative_sizes`
        if isinstance(self.dataset_meta, Sequence):
            dataset_slices = self.dataset_meta[0]['cumulative_sizes']
            if not self._setups:
                self._setups = True
                for dataset_meta, metric in zip(self.dataset_meta,
                                                self.metrics):
                    metric.dataset_meta = dataset_meta
        else:
            dataset_slices = self.dataset_meta['cumulative_sizes']
        return dataset_slices

    def evaluate(self, size: int) -> dict:
        """Invoke ``evaluate`` method of each metric and collect the metrics
        dictionary.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.

        Returns:
            dict: Evaluation results of all metrics. The keys are the names
            of the metrics, and the values are corresponding results.
        """
        metrics_results = OrderedDict()
        dataset_slices = self._get_cumulative_sizes()
        logger: MMLogger = MMLogger.get_current_instance()
        assert len(dataset_slices) == len(self.dataset_prefixes)

        for dataset_prefix, start, end, metric in zip(
                self.dataset_prefixes, [0] + dataset_slices[:-1],
                dataset_slices, self.metrics):
            if len(metric.results) == 0:
                warnings.warn(
                    f'{metric.__class__.__name__} got empty `self.results`.'
                    'Please ensure that the processed results are properly '
                    'added into `self.results` in `process` method.')

            results = collect_results(metric.results, size,
                                      metric.collect_device)

            if is_main_process():
                # cast all tensors in results list to cpu
                results = _to_cpu(results)
                _metrics = metric.compute_metrics(
                    results[start:end])  # type: ignore

                if metric.prefix:
                    final_prefix = '/'.join((dataset_prefix, metric.prefix))
                else:
                    final_prefix = dataset_prefix
                logger.info(f'================{final_prefix}================')
                metric_results = {
                    '/'.join((final_prefix, k)): v
                    for k, v in _metrics.items()
                }

                # Check metric name conflicts
                for name in metric_results.keys():
                    if name in metrics_results:
                        raise ValueError(
                            'There are multiple evaluation results with '
                            f'the same metric name {name}. Please make '
                            'sure all metrics have different prefixes.')
                metrics_results.update(metric_results)
            metric.results.clear()
        
        if is_main_process():
            # per task performance
            pertask_metric = defaultdict()
            pertask_mAP_key = []
            pertask_mAP_value = []
            original_metric = []
            for key, value in metrics_results.items():
                metric_type = key.split('/')
                if len(metric_type) > 3:   
                    original_metric.append(key)
                    task_name =  metric_type[2]
                    if task_name in self.evaluation_tasks:
                        metric_type[2] = metric_type[1]
                        metric_type[1] = task_name
                        new_metric_type = '/'.join(metric_type)
                        pertask_metric[new_metric_type] = value
                        if metric_type[-1] == 'bbox_mAP':
                            pertask_mAP_key.append(task_name)
                            pertask_mAP_value.append(value)
            
            for key in original_metric:
                metrics_results.pop(key)

            logger.info('Task: ' + '  '.join(pertask_mAP_key))
            logger.info('bbox_mAP: ' + '  '.join([f'{v:.3f}' for v in pertask_mAP_value]))
            logger.info('================pertask_performance================')        
            
            # avg results of natural/remote/underwater tasks
            avg_metric = defaultdict(list)
            for key, value in metrics_results.items():
                metric_type = key.split('/', 1)[-1]
                if value > 0.0:
                    avg_metric[metric_type].append(value)

            avg_value_list = []
            for metric_type, values in avg_metric.items():
                avg_value = sum(values) / len(values)
                metrics_results[f'avg/{metric_type}'] = avg_value
                avg_value_list.append(avg_value)
            
            logger.info(f'bbox_mAP_copypaste: {avg_value_list[0]:.3f} '
                        f'{avg_value_list[1]:.3f} {avg_value_list[2]:.3f} {avg_value_list[3]:.3f} '
                        f'{avg_value_list[4]:.3f} {avg_value_list[5]:.3f}')
            logger.info('================avg_performance================')
            
            metrics_results.update(pertask_metric)

            metrics_results = [metrics_results]
        else:
            metrics_results = [None]  # type: ignore

        broadcast_object_list(metrics_results)
        return metrics_results[0]