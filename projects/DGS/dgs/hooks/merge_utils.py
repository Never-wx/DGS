import torch
import copy
import torch.nn as nn
import re
from collections import defaultdict, OrderedDict
from typing import Dict, Union, Tuple, List, Any, Literal, Optional

def reparameter(model, delete=False):
    # for module in model.modules():
    #     if hasattr(module, '__rep__'):
    #         module.__rep__(delete=delete)  
    if not hasattr(model, '_rep_modules'):
        model._rep_modules = [m for m in model.modules() if hasattr(m, '__rep__')]
    # 直接遍历缓存的 rep_modules
    for module in model._rep_modules:
        module.__rep__(delete=delete)  

def load_ckpt(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # get state_dict from checkpoint
    if isinstance(checkpoint, OrderedDict):
        checkpoint = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    return checkpoint

def pop_keys_prefix(source_dict: Dict, prefix: str) -> Dict:
    """
    弹出以 prefix 开头的键值对，并返回这些被弹出的项。
    用于分离不需要参与 merge 的冻结层。
    """
    extracted_dict = {}
    keys = list(source_dict.keys())
    for key in keys:
        if key.startswith(prefix):
            extracted_dict[key] = source_dict.pop(key)
    return extracted_dict

def pop_keys_suffix(source_dict: Dict, suffix: str) -> Dict:
    """
    弹出以 suffix 开头的键值对，并返回这些被弹出的项。
    用于分离不需要参与 merge 的冻结层。
    """
    extracted_dict = {}
    keys = list(source_dict.keys())
    for key in keys:
        if key.endswith(suffix):
            extracted_dict[key] = source_dict.pop(key)
    return extracted_dict

def pop_specific_keys(source_dict: Dict, keys_to_remove: Union[str, List[str]]):
    """
    弹出指定的键 (Exact match)。
    """
    if isinstance(keys_to_remove, str):
        keys_to_remove = [keys_to_remove]
    
    for key in keys_to_remove:
        if key in source_dict:
            source_dict.pop(key)

def extract_containing_keys(source_dict: Dict, keyword: str) -> Dict:
    """
    提取包含特定 keyword 的键值对（例如 'lora_'），不从原字典删除，只提取。
    """
    return {k: v for k, v in source_dict.items() if keyword in k}

def state_dict_to_vector(state_dict, remove_keys=[]):
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
    )

def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    # create a reference dict to define the order of the vector
    reference_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))
    
    # create a shared state dict using the refence dict
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())
    
    # add back the encoder and decoder embedding weights.
    if "transformer.shared.weight" in sorted_reference_dict:
        for key in remove_keys:
            sorted_reference_dict[key] = sorted_reference_dict[
                "transformer.shared.weight"
            ]
    return sorted_reference_dict


def check_parameterNamesMatch(checkpoints):
    parameter_names = set(checkpoints[0].keys())

    if len(checkpoints) >= 2:
        # raise ValueError("Number of models is less than 2.")
        for checkpoint in checkpoints[1:]:
            current_parameterNames = set(checkpoint.keys())
            if current_parameterNames != parameter_names:
                raise ValueError(
                    "Differing parameter names in models. "
                    f"The different parameters are {parameter_names.symmetric_difference(current_parameterNames)}"
                )

def check_state_dicts_equal(state_dict1, state_dict2):
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        return False

    for key in state_dict1.keys():
        if not torch.equal(state_dict1[key], state_dict2[key]):
            return False

    return True

def extract_expert_state_dicts(state_dict):
    """
    Extract expert weights from checkpoint organized by expert index first.
    
    Args:
        checkpoint (dict): Model checkpoint dictionary containing state_dict
    
    Returns:
        list: List of state dictionaries, one for each expert index
    """
    # Dictionary structure: expert_idx -> layer_prefix -> param_name -> tensor
    expert_params = defaultdict(lambda: defaultdict(dict))
    
    # Regular expression to match expert parameters
    pattern = r'(.+?)\.experts\.(\d+)\.(.+)'
    
    # Find max expert index
    max_expert_idx = 0

    expert_key = []
    for key, value in state_dict.items():
        if 'experts' in key:
            expert_key.append(key)
    
    # Collect all expert weights
    for key in expert_key:
        match = re.match(pattern, key)
        if match:
            layer_prefix = match.group(1)  # e.g., 'backbone.layers.0'
            expert_idx = int(match.group(2))  # e.g., 0, 1, 2
            param_name = match.group(3)  # e.g., 'lora_A', 'weight'
            
            # Store the parameter in the nested dictionary
            expert_params[expert_idx][layer_prefix][param_name] = state_dict.pop(key)
            
            # Track maximum expert index
            max_expert_idx = max(max_expert_idx, expert_idx)
    
    # Convert the nested dictionary to a list of state dictionaries
    expert_state_dicts = []
    
    for expert_idx in range(max_expert_idx + 1):
        if expert_idx in expert_params:
            # Create state dict for this expert
            expert_dict = {}
            
            for layer_prefix, params in expert_params[expert_idx].items():
                for param_name, tensor in params.items():
                    # Standardize key format with expert index 0
                    key = f"{layer_prefix}.experts.0.{param_name}"
                    expert_dict[key] = tensor
            
            expert_state_dicts.append(expert_dict)
        else:
            # Handle missing expert indices by appending an empty dict
            expert_state_dicts.append({})
    
    return expert_state_dicts






# def extract_stacked_expert_weights(checkpoint):
#     """
#     Extract expert weights from checkpoint and stack them by layer.
    
#     For each layer and parameter (e.g., lora_A, lora_B), stacks weights from all experts
#     along dimension 0 to create a single tensor of shape [num_experts, ...].
    
#     Args:
#         checkpoint (dict): Model checkpoint dictionary containing state_dict
    
#     Returns:
#         dict: Dictionary with modified keys mapping to stacked expert weights
#     """
#     # Get state dict from checkpoint
#     if 'state_dict' in checkpoint:
#         state_dict = checkpoint['state_dict']
#     else:
#         state_dict = checkpoint
    
#     # Dictionary to organize weights by layer and parameter type
#     # Structure: {layer_prefix: {param_name: {expert_idx: tensor}}}
#     layer_params = defaultdict(lambda: defaultdict(dict))
    
#     # Regular expression to match expert parameters
#     pattern = r'(.+?)\.experts\.(\d+)\.(.+)'
    
#     # Collect expert weights
#     expert_key = []
#     for key, value in state_dict.items():
#         if 'experts' in key:
#             expert_key.append(key)

#     for key in expert_key:
#         match = re.match(pattern, key)
#         if match:
#             layer_prefix = match.group(1)  # e.g., 'backbone.layers.0'
#             expert_idx = int(match.group(2))  # e.g., 0, 1, 2
#             param_name = match.group(3)  # e.g., 'lora_A', 'lora_B', 'weight'
            
#             layer_params[layer_prefix][param_name][expert_idx] = state_dict.pop(key)
    
#     # Create stacked tensors and modified keys
#     experts_dict_list = []
#     experts_dict = {}

#     for layer_prefix, params in layer_params.items():
#         for param_name, expert_tensors in params.items():
#             # Get max expert index to ensure we include all experts
#             max_expert_idx = max(expert_tensors.keys())
            
#             # Create list of tensors in order from expert 0 to max
#             tensor_list = []
#             for expert_idx in range(max_expert_idx + 1):
#                 if expert_idx in expert_tensors:
#                     tensor_list.append(expert_tensors[expert_idx])
            
#             # Stack tensors along dimension 0
#             stacked_tensor = torch.stack(tensor_list, dim=0)
            
#             # Create modified key for stacked tensor
#             modified_key = f"{layer_prefix}.experts.0.{param_name}"
#             experts_dict[modified_key] = stacked_tensor
    
#     return experts_dict



## TIES MERGING UTILS
def scan_checkpoints(work_dir, task_id=None):
    """
    Scan checkpoints from task_0 to task_id in work_dir.
    Expects structure like work_dir/coco_task_0/epoch_12.pth 
    """
    import os
    import glob
    
    available_dirs = [d for d in os.listdir(work_dir) if os.path.isdir(os.path.join(work_dir, d))]
    
    task_dirs = []
    if task_id is None:
        # return all task_i directories (sorted by the integer part)
        task_info = []
        for d in available_dirs:
            if '_task_' in d:
                try:
                    idx = int(d.split('_task_')[-1])
                    task_info.append((idx, d))
                except: continue
        task_info.sort()
        task_dirs = [os.path.join(work_dir, d) for idx, d in task_info]
    else:
        for i in range(task_id + 1):
            found = False
            # Find directory that ends with _task_{i}
            for d in available_dirs:
                if d.endswith(f'_task_{i}'):
                    task_dirs.append(os.path.join(work_dir, d))
                    found = True
                    break
            if not found:
                # If path exists as task_{i}, also try it
                if os.path.isdir(os.path.join(work_dir, f'task_{i}')):
                     task_dirs.append(os.path.join(work_dir, f'task_{i}'))

    checkpoints = []
    for d in task_dirs:
        ckpt = os.path.join(d, 'epoch_12.pth')
        if os.path.exists(ckpt):
            checkpoints.append(ckpt)
        else:
            alternate_ckpts = sorted(glob.glob(os.path.join(d, '*.pth')))
            if alternate_ckpts:
                # Use the latest epoch if possible, otherwise use the first one
                checkpoints.append(alternate_ckpts[-1])
    
    return checkpoints