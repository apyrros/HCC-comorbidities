import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import io

from train.train_consts import *


def save_checkpoint(state: dict, model_state: dict, isbest: bool, checkpoint: str):
    """
    Save checkpoint
    :param state: dict with keys: "epoch" - epoch num, "state_dict" - model, "optim_dict" - optimizer
    :param model_state: like previous, but with "state_dict" key only
    :param is_best: True if it is the best epoch in terms of loss
    :checkpoint: directory in the local file system for saving checkpoints
    """
    filepath = os.path.join(checkpoint, f'last_{EXPERIMENT_NAME}.pth')
    model_filepath = os.path.join(checkpoint, f'model_last_{EXPERIMENT_NAME}.pth')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.makedirs(checkpoint)

    torch.save(state, filepath)
    torch.save(model_state, model_filepath)
    if isbest:
        print("Saving best path")
        shutil.copyfile(filepath, os.path.join(checkpoint, f'best_{EXPERIMENT_NAME}.pth'))
        shutil.copyfile(model_filepath, os.path.join(checkpoint, f'model_best_{EXPERIMENT_NAME}.pth'))


def load_checkpoint(checkpoint: str, model: nn.Module, optimizer: optim.Optimizer = None):
    """
    Load model and optimizer checkpoints
    :param checkpoint: path in the local file system to checkpoints
    :param model: model for loading state dict
    :param optimizer: optimizer for loading state dict
    """
    if not os.path.exists(checkpoint):
        raise IOError("File doesn't exist {}".format(checkpoint))

    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint)
    else:
        checkpoint = torch.load(checkpoint, map_location='cpu')

    state_dict = {}
    for key in checkpoint['state_dict'].keys():
        if 'layers.0.' in key:
            state_dict[key.split('0.')[0].split('module.')[1] + key.split('0.')[1]] = checkpoint['state_dict'][key]
        elif 'layers.1.' in key:
            state_dict[key.replace('1', '8').split('module.')[1]] = checkpoint['state_dict'][key]
        elif 'module.' in key:
            state_dict[key.split('module.')[1]] = checkpoint['state_dict'][key]
        elif 'head.layers.8' in key:
            continue
        else:
            state_dict[key] = checkpoint['state_dict'][key]
    model.load_state_dict(state_dict, strict=False)

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])
        
        
def export_to_onnx(model, imgs):
    
    f = io.BytesIO()
    f.name = 'model.onnx'
    torch.onnx.export(model,  # model being run
                      [img.to(model.device) for img in imgs] if isinstance(imgs, list) else imgs.to(model.device),  # model input (or a tuple for multiple inputs)
                      f,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})
    return f

