import torch.nn as nn
import timm
from train.model.base_model import BaseModel
from train.train_consts import *
from torch.optim import lr_scheduler



class TimmSwin(BaseModel):
    def __init__(self, model_arch='swin_base_patch4_window12_384', pretrain=True, model_name='',
                 **kwargs):
        super().__init__(model_name=f'Timm_{model_arch}_{model_name}', pretrain=False,
                         **kwargs)
        self.model = timm.create_model(model_arch, pretrain=pretrain)
        n_features = self.model.head.in_features
        self.head = nn.Linear(n_features, len(CONDITIONS))
        self.init_()
        
        self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optim, T_0=5, T_mult=1, eta_min=1e-6, verbose=True)
        
    # def forward(self, x):
    #     out = self.model(x)
    #     out = self.classifier(out)
    #     return out
    
    
class TimmEfficientNetV2(BaseModel):
    def __init__(self, model_arch='efficientnetv2_s', pretrain=False, model_name='',
                 **kwargs):
        super().__init__(model_name=f'Timm_{model_arch}_{model_name}', pretrain=False,
                         **kwargs)
        self.model = timm.create_model(model_arch, pretrained=pretrain)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, len(CONDITIONS))
        self.init_()
        
        self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optim, T_0=5, T_mult=1, eta_min=1e-6, verbose=True)
        

        