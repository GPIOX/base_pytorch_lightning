import lightning as L
import pytorch_lightning as pl
from torch import nn, optim
import numpy as np
from model.resnet import resnet
from model import base_loss
import torch



class ResNetTrainer(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        # print(self.kwargs)
        self.net = getattr(resnet, self.kwargs['name'])(**self.kwargs['model_arg'])
        self.sync_dist=True if self.kwargs['trainer']['device'] > 1 else False
        self.loss = getattr(base_loss, self.kwargs['trainer']['trainer_arg']['loss']['name'])(**self.kwargs['trainer']['trainer_arg']['loss'])


    def forward(self, *args, **kwargs):
        pass

    def training_step(self, *args, **kwargs):
        loss, acc = self._forward_pass(*args, **kwargs)

        return {
            'loss': loss,
            'acc': acc
        }
    
    def validation_step(self, *args, **kwargs):
        loss, acc = self._forward_pass(*args, **kwargs)

        return {
            'loss': loss,
            'acc': acc
        }
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        
        self.log('train_loss', avg_loss)
        self.log('train_acc', avg_acc)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        
        self.log('valid_loss', avg_loss)
        self.log('valid_acc', avg_acc, prog_bar=True)

    def _forward_pass(self, *args, **kwargs):
        img, label, _ = args[0] # args[0]包含了image、label、index
        pred = self.net(img)

        loss = self.loss(pred, label)
        loss = loss.mean()

        pred = pred.argmax(dim=1)
        acc = (pred == label).float().sum()

        return loss, acc

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.args['lr'])
        optimizer_dict = self.kwargs['trainer']['trainer_arg']['optimizer']
        scheduler_dict = self.kwargs['trainer']['trainer_arg']['lr_scheduler']

        optimizer = getattr(optim, optimizer_dict['name'])(self.parameters(), **optimizer_dict['optimizer_arg'])
        
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args['lr_step'], gamma=self.args['gamma'])
        scheduler = getattr(optim.lr_scheduler, scheduler_dict['name'])(optimizer, **scheduler_dict['scheduler_arg'])


        return [optimizer], [scheduler]



