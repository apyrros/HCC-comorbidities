import torch
import torch.nn as nn

classifier_criterion = nn.BCELoss()
regression_criterion = nn.MSELoss()

def multilabel_regression_loss(y_pred, y_label):
    pred_conditions = y_pred[:,:-2]
    pred_scores = y_pred[:, -2:]
    label_conditions = y_label[:,:-2]
    label_scores = y_label[:, -2:]

    loss = classifier_criterion(pred_conditions, label_conditions)
    loss += regression_criterion(pred_scores, label_scores)
    return loss

