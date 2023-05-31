import torch.nn as nn
from train.model.base_model import BaseModel
from train.utils.checkpoints import load_checkpoint
import torchxrayvision as xrv
import contextlib
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import mlflow
from train.train_consts import *
import torch.nn.functional as F



class XRVResnet50(BaseModel):
    def __init__(self, model_name='XRVResnet50', mixed_precision=True, **kwargs):
        """
        """
        super().__init__(model_name=model_name, mixed_precision=mixed_precision, **kwargs)
        self.model = xrv.models.ResNet(weights="resnet50-res512-all").model
        self.model.fc = nn.Linear(self.model.fc.in_features, len(CONDITIONS))
        
        
        self.init_()

        
class XRVDensenet121Frozen(BaseModel):
    def __init__(self, model_name='XRVDensenet121Frozen', mixed_precision=True, **kwargs):
        """
        """
        super().__init__(model_name=model_name, mixed_precision=mixed_precision, **kwargs)
        self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.model.op_threshs = None # prevent pre-trained model calibration
        
        for param in list(self.model.parameters()):
            param.requires_grad = False
    
        self.fc =  nn.Sequential(
            nn.Linear(1024, 256),
            nn.SELU(),
            nn.Linear(256, 512),
            nn.SELU(),
            nn.Linear(512, len(CONDITIONS))
        )
        
        self.init_()
        
    def features2(self, x):
        
        features = self.model.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        return out
        
    def forward(self, x):
        features = self.features2(x)
        out = self.fc(features)
        
        return out
        
        
        
class XRVDensenet121NIH(BaseModel):
    def __init__(self, model_name='XRVDensenet121JF', mixed_precision=True, **kwargs):
        """
        """
        super().__init__(model_name=model_name, mixed_precision=mixed_precision, **kwargs)
        model = xrv.models.DenseNet(weights="densenet121-res224-nih")
        model.op_threshs = None # prevent pre-trained model calibration
        model.classifier = torch.nn.Linear(1024, len(CONDITIONS)) # reinitialize classifier
        self.model = model
        self.init_()
        
        
class XRVDensenet121(BaseModel):
    def __init__(self, model_name='XRVDensenet121Chex', 
                 weights='densenet121-res224-chex',
                 pretrain=False, mixed_precision=True, **kwargs):
        """
        """
        super().__init__(model_name=model_name, mixed_precision=mixed_precision, pretrain=pretrain, **kwargs)
        model = xrv.models.DenseNet(weights=weights)
        model.op_threshs = None # prevent pre-trained model calibration
        model.classifier = torch.nn.Linear(1024, len(CONDITIONS)) # reinitialize classifier
        self.model = model
        
        self.init_()
        
    def features2(self, x):
        
        features = self.model.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        return out
    
    def forward(self, x):
        features = self.features2(x)
        out = self.model.classifier(features)
        
        return out
    
    
class XRVHybrid(BaseModel):
    def __init__(self, model_name='Hybrid', 
                 weights='densenet121-res224-chex',
                 pretrain=False, mixed_precision=True, **kwargs):
        """
        """
        super().__init__(model_name=model_name, mixed_precision=mixed_precision, pretrain=pretrain, **kwargs)
        self.model2 = XRVDensenet121(weights=weights)
        
        
        self.init_()
        
    def forward(self, x):
        res = self.model(x)
        res2 = self.model2(x[:, :1, :, :])
        
        return res + res2
    
class XRVAllChannels(BaseModel):
    def __init__(self, model_name='XRVAllChannels', 
                 weights='densenet121-res224-chex',
                 pretrain=False, mixed_precision=True, **kwargs):
        """
        """
        super().__init__(model_name=model_name, mixed_precision=mixed_precision, pretrain=pretrain, **kwargs)
        self.model = None
        self.model1 = XRVDensenet121(weights=weights)
        self.model2 = XRVDensenet121(weights=weights)
        self.model3 = XRVDensenet121(weights=weights)
        
        
        self.init_()
        
    def forward(self, x):
        res1 = self.model1(x[:, :1, :, :])
        res2 = self.model2(x[:, 1:2, :, :])
        res3 = self.model2(x[:, 2:, :, :])
        
        return res1 + res2 + res3


        
    
    
        




