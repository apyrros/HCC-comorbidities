import os
import shutil
import torch

# Functions in this file are inspired by the following:
# https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/utils.py

def save_checkpoint(state, model_state, isbest, checkpoint):
    """
    Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    @param state     : contains model's state_dict, may contain other keys such as epoch, optimizer state_dict (dict)
    @param isbest   : True if it is the best model seen till now (bool)
    @param checkpoint: folder where parameters are to be saved (string)
    """
    filepath = os.path.join(checkpoint, 'last.pth')
    model_filepath = os.path.join(checkpoint, 'model_last.pth')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.makedirs(checkpoint)

    torch.save(state, filepath)
    torch.save(model_state, model_filepath)
    if isbest:
        print("Saving best path")
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth'))
        shutil.copyfile(model_filepath, os.path.join(checkpoint, 'model_best.pth'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """
    Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    @param checkpointdir: directory with checkpoint files (string)
    @param model        : model for which the parameters are loaded (DeepConvNet)
    @param optimizer    : resume optimizer from checkpoint (optim)
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
        else:
            state_dict[key] = checkpoint['state_dict'][key]
    model.load_state_dict(state_dict)

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

