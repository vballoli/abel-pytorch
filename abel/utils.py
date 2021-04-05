import torch

from typing import Iterable

def get_weight_norm(param_groups: Iterable) -> torch.Tensor:
    """
    Returns weight norm of the param groups
    
    Args:
        param_groups (Iterable): List of parameters of the model
    """
    norm = None
    for group in param_groups:
        for p in group['params']:
            if norm is None:
                norm = torch.norm(p, 2) ** 2
            else:
                norm += torch.norm(p, 2) ** 2
                
    return norm
