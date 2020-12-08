import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
#from tqdm import tqdm

from dataset import ClassifierDataset

parser = argparse.ArgumentParser(description="Calculate mean and std of dataset")
parser.add_argument("path", help="Path to dataset")
args = parser.parse_args()
path = args.path

train_dataset = ClassifierDataset(path, size=256, train=True, age_norm=100, raf_norm=10)
test_dataset = ClassifierDataset(path, size=256, train=False, age_norm=100, raf_norm=10)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=8)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8)

train_mean = 0.0
train_std = 0.0
test_mean = 0.0
test_std = 0.0

for img, label in train_dataloader:
    img = img.numpy()
    train_mean += np.mean(img)
    train_std += np.std(img)

train_mean /= len(train_dataset)
train_std /= len(train_dataset)
print('Training mean: {}'.format(train_mean))
print('Training std: {}'.format(train_std))

for img, label in test_dataloader:
    img = img.numpy()
    test_mean += np.mean(img)
    test_std += np.std(img)

test_mean /= len(test_dataset)
test_std /= len(test_dataset)
print('Testing mean: {}'.format(test_mean))
print('Testing std: {}'.format(test_std))

