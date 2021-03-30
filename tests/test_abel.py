import torch
from torch import optim
from abel import ABEL

from torchvision.models import resnet18

def test_abel():
    model = resnet18()
    optimizer = optim.SGD(model.parameters(), 1e-3)
    scheduler = ABEL(optimizer, 0.1)

    for module in model.modules():
        if hasattr(module, 'weight'):
            module.weight.data.copy_(torch.ones_like(module.weight))
            
    scheduler.step()
    
    for module in model.modules():
        if hasattr(module, 'weight'):
            module.weight.data.copy_(5 * torch.ones_like(module.weight))
            
    scheduler.step()
    
    for module in model.modules():
        if hasattr(module, 'weight'):
            module.weight.data.copy_(25 * torch.ones_like(module.weight))
            
    scheduler.step()
    
    for module in model.modules():
        if hasattr(module, 'weight'):
            module.weight.data.copy_(-0.01 * torch.ones_like(module.weight))
            
    scheduler.step()
    
    for module in model.modules():
        if hasattr(module, 'weight'):
            module.weight.data.copy_(-1. * torch.ones_like(module.weight))
            
    scheduler.step()
    
    assert scheduler._get_closed_form_lr()[0] < 1e-3
