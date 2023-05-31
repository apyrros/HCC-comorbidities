import torch.nn as nn
from train.model.resnet34 import ResNet34
from train.model.cc_resnet_old import resnet34
from train.utils.checkpoints import load_checkpoint
import numpy as np
import contextlib
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import mlflow
from train.train_consts import *


class BaseModel(nn.Module):
    def __init__(self, pretrain: bool = PRETRAIN, model_name='Baseline',
                 mixed_precision=True, lr=1e-3, lr_factor=0.5, lr_patience=10, lr_threshold=0.0001, lr_min=1e-5,
                pos_weight=None, load_model: bool = True):    
        """
        """
        super().__init__()
        if load_model:
            if pretrain:
                self.model = resnet34(num_classes=len(CONDITIONS), pretrained=False)
                load_checkpoint(PRETRAIN, self.model)
            else:
                self.model = resnet34(pretrained=False, num_classes=len(CONDITIONS))
        
        self.pretrain = pretrain
        self.model_name = model_name
        self.mixed_precision = mixed_precision
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.lr_threshold = lr_threshold
        self.pos_weight = pos_weight

        self.classifier_criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduce=False
                                                        )
        self.regression_criterion = nn.MSELoss(reduce=False)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.init_()
        
    def init_(self):
        self.optim = optim.AdamW(self.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optim, 'min', factor=self.lr_factor, patience=self.lr_patience,
                                           verbose=True, threshold=self.lr_threshold)

        
        self.scaler = torch.cuda.amp.GradScaler()
        self.to(self.device)

        self.mode = 'train'

    def loss_fct(self, y_pred: torch.Tensor, y_label: torch.Tensor) -> torch.Tensor:
        """
        Calculate full loss for multitask (binary classifications + regressions)
        :param y_pred: tensor with predictions (after sigmoid for clasification)
        :param y_label: tensor with ground truth labels
        :return: tensor with total loss value
        """
        #y_label[torch.isnan(y_label)] = y_pred[torch.isnan(y_label)].to(y_label)
        
        class_ids = [i for i, cond_type in enumerate(CONDITIONS.values()) if cond_type == CLASS_TASK]
        reg_ids = [i for i, cond_type in enumerate(CONDITIONS.values()) if cond_type == REG_TASK]
        pred_conditions = y_pred[:, class_ids]
        pred_scores = y_pred[:, reg_ids]
        label_conditions = y_label[:, class_ids]
        label_scores = y_label[:, reg_ids]
        
        class_loss = self.classifier_criterion(pred_conditions + 1e-5, label_conditions)
        # false_positives = (pred_conditions > 0.5) & (label_conditions < 0.5)
        # class_loss[false_positives] *= 0.7
        class_weights = [COND_WEIGHTS.get(cond, 1.) for cond, cond_type in CONDITIONS.items() if cond_type == CLASS_TASK]
    
        class_loss = torch.mul(class_loss, torch.Tensor(class_weights).to(self.device))
        
        reg_loss = self.regression_criterion(pred_scores, label_scores)
        reg_weights = [COND_WEIGHTS.get(cond, 1.) for cond, cond_type in CONDITIONS.items() if cond_type == REG_TASK]
        reg_loss = torch.mul(reg_loss, torch.Tensor(reg_weights).to(self.device))
        
        loss = class_loss.mean() 
        if len(reg_ids) > 0:
            loss += reg_loss.mean()

        if torch.isnan(loss):
            return torch.Tensor([torch.nan])
        return loss

    def log_params(self):
        mlflow.log_param('lr', self.lr)
        mlflow.log_param('lr_min', self.lr_min)
        mlflow.log_param('mixed_precision', self.mixed_precision)
        mlflow.log_param('lr_factor', self.lr_factor)
        mlflow.log_param('lr_patience', self.lr_patience)
        mlflow.log_param('lr_threshold', self.lr_threshold)
        mlflow.log_param('pretrain', self.pretrain)
        mlflow.log_param('pos_weight', self.pos_weight)
        mlflow.log_param('scheduler', self.scheduler)
        mlflow.log_param('optim', self.optim)
        

    def forward(self, x):
        x = self.model(x.to(self.device))

        return x

    def train_step(self, x, labels):
        if self.mode != 'train':
            self.train()
            self.mode = 'train'

        with torch.cuda.amp.autocast() if self.mixed_precision else contextlib.nullcontext():
            for param in self.parameters():
                param.grad = None

            preds = self.forward(x)
            loss = self.loss_fct(preds, labels.to(self.device))

            if not torch.isnan(loss):
                if self.mixed_precision:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optim.step()

        return loss.detach().cpu().numpy()

    def predict(self, x, labels=None, raw_out=False):
        if self.mode != 'eval':
            self.eval()
            self.mode = 'eval'

        loss = None
        with torch.no_grad():
            logits = self.forward(x)
            if labels is not None:
                loss = self.loss_fct(logits, labels.to(self.device))

        if loss is not None:
            loss = loss.detach().cpu().numpy()
        
        class_ids = [i for i, cond_type in enumerate(CONDITIONS.values()) if cond_type == CLASS_TASK]
        logits_sigmoid = logits.clone()
        logits_sigmoid[:, class_ids] = torch.sigmoid(logits[:, class_ids])
        return loss, logits_sigmoid.detach().cpu() if not raw_out else logits


