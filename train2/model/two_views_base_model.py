import torch.nn as nn
from train.model.base_model import BaseModel
from train.model.cc_resnet_old import Head, resnet34
from train.utils.checkpoints import load_checkpoint
from train.train_consts import *


class TwoViewsBaseModel(BaseModel):
    def __init__(self, model_name='TwoViewsBaseLine', **kwargs):    
        super().__init__(model_name=model_name, **kwargs)
        self.model_front = resnet34(num_classes=len(CONDITIONS), pretrained=False, add_head=False)
        self.model_lat = resnet34(num_classes=len(CONDITIONS), pretrained=False, add_head=False)
        self.head = Head(in_planes=1024 * 2, mid_planes=512, out_planes=len(CONDITIONS))
        if self.pretrain:
            load_checkpoint(PRETRAIN, self.model)
        self.init_()

    def forward(self, x):
        channels_count = int(x.size()[1] / 2)
        x_front, x_lat = x[:, :channels_count, :, :], x[:, channels_count:, :, :]
        x_front = self.model_front(x_front.to(self.device))
        x_lat = self.model_lat(x_lat.to(self.device))
        return self.head(torch.cat([x_front, x_lat], dim=1))
    
    def log_params(self):
        super().log_params()
        mlflow.log_param('two_views', True)
