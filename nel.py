from torch.nn.modules import Module
import torch
import torch.nn as nn
import torch.nn.functional as F

class NEL_Loss(Module):
    def __init__(self):
        super(NEL_Loss, self).__init__()

        
    def forward(self, target, prediction):
        prediction = prediction.requires_grad_()
        C1 = 1E-8
        Eudis = torch.sum(target * target) + C1
        norm = torch.sqrt(Eudis)
        target = target /  norm 
        prediction =  prediction / norm
        delta = prediction - target 
        loss = torch.sum( delta * delta) 
        return loss
        
if __name__ == '__main__':
    criterion = NEL_Loss()
    a = torch.abs(torch.randn(2,1,2,2))
    b = a * (2)
    loss = criterion(a,b)
    loss.backward()
    print(loss)
