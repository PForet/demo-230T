import torch
from torch import nn
from torch.nn import functional as F

class GenderClassifier(nn.Module):
    def __init__(self):
        super(GenderClassifier, self).__init__()
        self.layer1 = nn.Linear(1280, 2)
    
    def forward(self, inputs):
        inputs = self.layer1(inputs)
        return F.log_softmax(inputs, dim=1)

class AgeClassifier(nn.Module):
    def __init__(self):
        super(AgeClassifier, self).__init__()
        self.layer1 = nn.Linear(1280, 9)
    
    def forward(self, inputs):
        inputs = self.layer1(inputs)
        return F.log_softmax(inputs, dim=1)

class ExpressionClassifier(nn.Module):
    def __init__(self, nfold=10):
        super(ExpressionClassifier, self).__init__()
        self.nfold = nfold
        for f in range(nfold):
            setattr(self, 'layer_{}'.format(f), nn.Linear(1280, 7))
    
    def forward(self, inputs):
        layer = getattr(self, 'layer_0')
        stack = F.softmax(layer(inputs), dim=1)
        for i in range(1, self.nfold):
            layer = getattr(self, 'layer_{}'.format(i))
            stack = torch.cat((stack, F.softmax(layer(inputs), dim=1)), dim=0)
        return stack.mean(0).view(1,-1)