import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms

class CustomLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(CustomLoss, self).__init__()
        self.loss = []
        del kwargs['name']

        # 记录好权重
        if kwargs['weight'] is None:
            self.weight = None
        del kwargs['weight']

        for k, v in kwargs.items():
            self.loss.append(
                getattr(nn, v['name'])(**v['loss_arg'])
            )

    def forward(self, pred, label, *args, **kwargs):
        loss_value = 0.0
        for index, loss in enumerate(self.loss):
            # self.weight不为None时表示各个损失函数进行加权
            loss_value += loss(pred, label) if self.weight is None else loss(pred, label) * self.weight[index]

        return  loss_value

