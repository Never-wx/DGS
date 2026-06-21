# Copyright (c) OpenMMLab. All rights reserved.
import copy
import random
import yaml
import warnings
import os.path as osp
import os
import json
from typing import List, Union
from collections import OrderedDict
from mmengine.fileio import get_local_path
from mmengine.logging import MMLogger
from mmdet.registry import DATASETS
from mmdet.datasets.api_wrappers import COCO
from mmdet.datasets.base_det_dataset import BaseDetDataset
from mmdet.datasets.coco import CocoDataset

@DATASETS.register_module()
class CocoIncDataset(BaseDetDataset):
    """Dataset for COCO."""
    def __init__(self, 
                 *args, 
                 start: int = 0, 
                 end: int = 39,
                 setting: str = 'cur_text',
                 **kwargs) -> None:

        self.start = start
        self.end = end
        self.setting = setting
        super().__init__(*args, **kwargs) 
        self._metainfo['start'] = start       
        self._metainfo['ori_classes'] = self.metainfo['classes'][:self.start]
        self._metainfo['new_classes'] = self.metainfo['classes'][self.start:]
    
    METAINFO = {
        'classes':
        ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
         'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
         'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
         'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
         'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
         'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
         'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
         'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
         'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
        
        'id':
        (1, 2, 3, 4, 5, 6, 7, 
         8, 9, 10, 11, 13, 
         14, 15, 16, 17, 18, 19, 20, 
         21, 22, 23, 24, 25, 27, 28,
         31, 32, 33, 34, 35, 36, 
         37, 38, 39, 40, 41, 
         42, 43, 44, 46, 47, 48, 
         49, 50, 51, 52, 53, 54, 55, 
         56, 57, 58, 59, 60, 61, 62, 
         63, 64, 65, 67, 70, 72, 
         73, 74, 75, 76, 77, 78, 
         79, 80, 81, 82, 84, 85, 86, 
         87, 88, 89, 90),     
    }

    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501

        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])
        
        if self.setting  == 'cur_text':     # partial text mapping
            self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}   
        else:   # full text mapping
            self.cat2label = {}
            for i, cat_id in enumerate(self.cat_ids): 
                index = self.metainfo['id'].index(cat_id)      
                self.cat2label.update({cat_id: index})

        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco
        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        if 'file_name' not in img_info.keys() and 'coco_url' in img_info.keys():
            file_name = img_info['coco_url'].replace(
                'http://images.cocodataset.org/', '')            
            img_info['file_name'] = file_name            
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['custom_entities'] = True
            data_info['caption_prompt'] = self.caption_prompt
            if self.setting == 'cur_text':  # partial text mapping
                data_info['text'] = self.metainfo['classes'][self.start:self.end]
                data_info['ori_text'] = self.metainfo['classes'][:self.start]
            else:   # full text mapping
                data_info['text'] = self.metainfo['classes'][:self.end]
                data_info['ori_text'] = self.metainfo['classes'][:self.start]

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        return data_info
    
    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        # obtain images that contain annotation
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos

@DATASETS.register_module()
class CDIOD_Agnostic_Dataset(CocoDataset):  
    def __init__(self, seen_tasks: str = None,  distn_cfg: dict = None, *args, **kwargs) -> None:
                              
        _, non_overlap_classes = self.datasets2clsnames(kwargs['metainfo'])
        kwargs['metainfo'] = dict(classes=non_overlap_classes)

        if distn_cfg is not None:  
            self.distn_cfg = distn_cfg
            if distn_cfg.type == 'distillation':
                seen_datasets, seen_classes = self.datasets2clsnames(seen_tasks)
                mapping_path = self.distn_cfg.get('task_id_mapping_path', None)
                prev_group_classes, group_classes = self.task2group(
                    seen_datasets, len(seen_datasets) - 1, mapping_path)
                self.ori_classes = prev_group_classes
                kwargs['metainfo'] = dict(classes=group_classes)
            
            elif distn_cfg.type == 'pseudo_labeling' or 'distillation_raw':  
                seen_datasets, seen_classes = self.datasets2clsnames(seen_tasks)
                prev_datasets = seen_datasets[:-1] if len(seen_datasets) > 1 else seen_datasets 
                _, prev_seen_classes = self.datasets2clsnames(prev_datasets)
                self.ori_classes = prev_seen_classes
                kwargs['metainfo'] = dict(classes=seen_classes)

        super().__init__(*args, **kwargs)  
    
    METAINFO = {
        'DIOR': ('Expressway-Service-area', 'Expressway-toll-station', 'airplane', 'airport', 'baseballfield', 
                'basketballcourt','bridge', 'chimney', 'dam', 'golffield', 'groundtrackfield', 'harbor', 
                'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'),
        'PascalVOC': ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person','pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
        'RUOD' : ('holothurian', 'echinus', 'scallop', 'starfish', 'fish', 'corals', 'diver', 'cuttlefish', 'turtle', 'jellyfish'),
        
        'DIOR_Task2-1': ('Expressway-Service-area', 'Expressway-toll-station', 'airplane', 'airport', 'baseballfield', 'basketballcourt','bridge', 'chimney', 'dam', 'golffield'),
        'DIOR_Task2-2': ('groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'),
        'PascalVOC_Task2-1': ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow'),
        'PascalVOC_Task2-2':  ('diningtable', 'dog', 'horse', 'motorbike', 'person','pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
             
        'DIOR_Task4-1': ('Expressway-Service-area', 'Expressway-toll-station', 'airplane', 'airport', 'baseballfield'),
        'DIOR_Task4-2': ('basketballcourt','bridge', 'chimney', 'dam', 'golffield'),
        'DIOR_Task4-3': ('groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium'),
        'DIOR_Task4-4': ('storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'),
        'PascalVOC_Task4-1': ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle'),
        'PascalVOC_Task4-2': ('bus', 'car', 'cat', 'chair', 'cow'),
        'PascalVOC_Task4-3':  ('diningtable', 'dog', 'horse', 'motorbike', 'person'),
        'PascalVOC_Task4-4': ('pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
        'RUOD_Task2-1' : ('holothurian', 'echinus', 'scallop', 'starfish', 'fish'), 
        'RUOD_Task2-2' : ('corals', 'diver', 'cuttlefish', 'turtle', 'jellyfish'),
        'coco_0-19': ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light'),
        'coco_20-39': ('fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow'),
        'coco_40-59': ('diningtable', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven'),
        'coco_60-79': ('toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
    }
    
    def datasets2clsnames(self, datasets):
        cls_names = ()
        if type(datasets) is not tuple:
            if type(datasets) is str:
                # Split space-separated strings into a list
                datasets = datasets.split(',')
            datasets = tuple(datasets)
        for dataset in datasets:
            cls_names += self.METAINFO[dataset]     
        non_overlap_classes = tuple(OrderedDict.fromkeys(cls_names))   
        return datasets, non_overlap_classes
    
    def task2group(self, datasets, cur_step, task_mapping_path=None):
        logger: MMLogger = MMLogger.get_current_instance()
        
        if task_mapping_path is None:
            if logger is not None and getattr(logger, 'log_file', None) is not None:
                work_dir = os.path.dirname(os.path.dirname(os.path.dirname(logger.log_file)))
            else:
                work_dir = 'work_dirs/CDIOD'
            
            task_mapping_path = os.path.join(work_dir, 'task_id_mapping.yaml')    
        
        with open(task_mapping_path, "r") as f:
            loaded_mapping = yaml.safe_load(f) 
        # Filter out task IDs beyond current task
        task_mapping = {k: v for k, v in loaded_mapping.items() if int(k) <= cur_step}
        
        groups = {}  # {group_id: [task_ids]}
        for k, v in task_mapping.items():
            if v not in groups:
                groups[v] = []
            groups[v].append(k)
        group_id = task_mapping[cur_step]
        
        group_classes = None

        if len(groups[group_id]) > 1:
            # keep only prev classes of current group
            group_datasets = [datasets[i] for i in groups[group_id]]
            _, prev_group_classes= self.datasets2clsnames(group_datasets[:-1])
            _, group_classes= self.datasets2clsnames(group_datasets)
        else:
            raise ValueError(f"group {group_id} size <= 1, can't generate pseudo_labels")
        
        return prev_group_classes, group_classes
        
    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.metainfo['classes'])
        self.cat_names = self.coco.load_cats(self.cat_ids)
        
        # mapping current label to global labelmap accroding to cls_name
        self.cat2label = {}
        for i, (cat_id, cat_name) in enumerate(zip(self.cat_ids, self.cat_names)): 
            index = self.metainfo['classes'].index(cat_name['name'])      
            self.cat2label.update({cat_id: index})
        
        is_test = self.ann_file.split('/')[3]
        if is_test == 'valid' or is_test == 'test':
            self.cat_ids = [i+1 for i in range(len(self.metainfo['classes']))]  # only for test set
        
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)
        
        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco
        return data_list
    
    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True
            data_info['ori_text'] = None

            if hasattr(self, 'distn_cfg'):
                # if self.distn_cfg.type == 'distillation':
                #     data_info['ori_text'] = self.ori_classes 
                # elif self.distn_cfg.type == 'pseudo_labeling':
                #     data_info['text'] = self.ori_classes 
                data_info['ori_text'] = self.ori_classes 
        
        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]
            
            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        return data_info

@DATASETS.register_module()
class CDIOD_Agnostic_Dataset_KD(CocoIncDataset):
    
    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501

        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])

        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        # mapping current label to global labelmap accroding to cls_name
        self.cat_names = self.coco.load_cats(self.cat_ids)   
        self.cat2label = {}
        for i, (cat_id, cat_name) in enumerate(zip(self.cat_ids, self.cat_names)): 
            index = self.metainfo['classes'].index(cat_name['name'])      
            self.cat2label.update({cat_id: index})
        
        is_test = self.ann_file.split('/')[3]
        if is_test == 'valid' or is_test == 'test':
            self.cat_ids = [i+1 for i in range(len(self.metainfo['classes']))]  # only for test set    


        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco
        return data_list

@DATASETS.register_module()
class ODinW13_Dataset(CocoDataset):   
    """Dataset for COCO."""

    METAINFO = {
        'AerialMaritimeDrone': ('boat', 'car', 'dock', 'jetski', 'lift'),
        'Aquarium' : ('fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray'),
        'CottontailRabbits' : ('Cottontail-Rabbit', ),
        'EgoHands': ('hand', ),
        'NorthAmericaMushroom' : ('CoW', 'chanterelle'),
        'Packages' : ('package', ),
        'PascalVOC' : ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                        'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                        'tvmonitor'),
        'pistols' : ('pistol', ),
        'pothole' : ('pothole', ),
        'Raccoon' : ('raccoon', ),
        'ShellfishOpenImages' : ('Crab', 'Lobster', 'Shrimp'),
        'thermalDogsAndPeople' : ('dog', 'person'),
        'VehiclesOpenImages' : ('Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck', )
    }

    def __init__(self, seen_tasks: str = None, distn_cfg: dict = None, *args, **kwargs) -> None:
        
        if distn_cfg is not None:  
            _, non_overlap_classes = self.datasets2clsnames(kwargs['metainfo'])
            kwargs['metainfo'] = dict(classes=non_overlap_classes)
            
            self.distn_cfg = distn_cfg
            if distn_cfg.type == 'distillation':  # gmoe distillation
                seen_datasets, seen_classes = self.datasets2clsnames(seen_tasks)
                mapping_path = self.distn_cfg.get('task_id_mapping_path', None)
                prev_group_classes, group_classes = self.task2group(
                    seen_datasets, len(seen_datasets) - 1, mapping_path)
                self.ori_classes = prev_group_classes
                kwargs['metainfo'] = dict(classes=group_classes)
            
            elif distn_cfg.type == 'pseudo_labeling' or 'distillation_raw':  # generate pseudo labels or raw distillation
                seen_datasets, seen_classes = self.datasets2clsnames(seen_tasks)
                prev_datasets = seen_datasets[:-1] if len(seen_datasets) > 1 else seen_datasets 
                _, prev_seen_classes = self.datasets2clsnames(prev_datasets)
                self.ori_classes = prev_seen_classes
                kwargs['metainfo'] = dict(classes=seen_classes)
        else:
            if 'metainfo' in kwargs and isinstance(kwargs['metainfo'], str):
                kwargs['metainfo'] = dict(classes=self.METAINFO[kwargs['metainfo']])

        super().__init__(*args, **kwargs) 

    def datasets2clsnames(self, datasets):
        cls_names = ()
        if type(datasets) is not tuple:
            if type(datasets) is str:
                # Split space-separated strings into a list
                datasets = datasets.split(',')
            datasets = tuple(datasets)
        for dataset in datasets:
            cls_names += self.METAINFO[dataset]     
        non_overlap_classes = tuple(OrderedDict.fromkeys(cls_names))   
        return datasets, non_overlap_classes
    
    def task2group(self, datasets, cur_step, task_mapping_path=None):
        logger: MMLogger = MMLogger.get_current_instance()
        
        if task_mapping_path is None:
            if logger is not None and getattr(logger, 'log_file', None) is not None:
                work_dir = os.path.dirname(os.path.dirname(os.path.dirname(logger.log_file)))
            else:
                work_dir = 'work_dirs/CDIOD'
            
            task_mapping_path = os.path.join(work_dir, 'task_id_mapping.yaml')    
        
        with open(task_mapping_path, "r") as f:
            loaded_mapping = yaml.safe_load(f) 
        # Filter out task IDs beyond current task
        task_mapping = {k: v for k, v in loaded_mapping.items() if int(k) <= cur_step}
        
        groups = {}  # {group_id: [task_ids]}
        for k, v in task_mapping.items():
            if v not in groups:
                groups[v] = []
            groups[v].append(k)
        group_id = task_mapping[cur_step]
        
        group_classes = None

        if len(groups[group_id]) > 1:
            # keep only prev classes of current group
            group_datasets = [datasets[i] for i in groups[group_id]]
            _, prev_group_classes= self.datasets2clsnames(group_datasets[:-1])
            _, group_classes= self.datasets2clsnames(group_datasets)
        else:
            raise ValueError(f"group {group_id} size <= 1, can't generate pseudo_labels")
        
        return prev_group_classes, group_classes

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True
            data_info['ori_text'] = None

            if hasattr(self, 'distn_cfg'):
                data_info['ori_text'] = self.ori_classes 
        
        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        return data_info


# @DATASETS.register_module()
# class ODinW13_Dataset(CocoDataset):   
#     """Dataset for COCO."""
#     def __init__(self, *args, **kwargs) -> None:
#         kwargs['metainfo'] = dict(classes=self.METAINFO[kwargs['metainfo']])
#         super().__init__(*args, **kwargs) 
    
#     METAINFO = {
#         'AerialMaritimeDrone': ('boat', 'car', 'dock', 'jetski', 'lift'),
#         'Aquarium' : ('fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray'),
#         'CottontailRabbits' : ('Cottontail-Rabbit', ),
#         'EgoHands': ('hand', ),
#         'NorthAmericaMushroom' : ('CoW', 'chanterelle'),
#         'Packages' : ('package', ),
#         'PascalVOC' : ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
#                         'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
#                         'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
#                         'tvmonitor'),
#         'pistols' : ('pistol', ),
#         'pothole' : ('pothole', ),
#         'Raccoon' : ('raccoon', ),
#         'ShellfishOpenImages' : ('Crab', 'Lobster', 'Shrimp'),
#         'thermalDogsAndPeople' : ('dog', 'person'),
#         'VehiclesOpenImages' : ('Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck', )
#     }

