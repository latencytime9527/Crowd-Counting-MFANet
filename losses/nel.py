from torch.nn.modules import Module
import torch
import torch.nn as nn
import torch.nn.functional as F

class NEL_Loss(Module):
    def __init__(self):
        super(NEL_Loss, self).__init__()

        
    def forward(self, target, prediction):
            prediction = prediction.requires_grad_()
            delta = target - prediction
            C = 1e-8
            error = torch.sum(torch.abs(delta))+C
            loss1 = torch.sum((delta * delta))
            dist = torch.sqrt(torch.sum(target * target))
            target = target / dist
            prediction = prediction / dist
            delta = target - prediction
            loss = torch.sum((delta * delta))
            return loss

if __name__ == '__main__':
    criterion = NEL_Loss()
    a = torch.abs(torch.randn(2,1,16,16))
    b = torch.abs(torch.randn(2,1,16,16)) 
    loss = criterion(a,b)
    loss.backward()
    print(loss)
