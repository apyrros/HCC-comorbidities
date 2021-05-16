#!/usr/bin/env python
# coding: utf-8
#
# Author:	Kazuto Nakashima
# URL:		http://kazuto1011.github.io
# Created:	2017-05-18

from __future__ import print_function

import copy
import os
import sys
import json

import click
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from PIL import Image

from torchvision import models, transforms
from matplotlib.colors import LinearSegmentedColormap

#'/home/user/example/parent/child'
current_path = os.path.abspath('.')

#'/home/user/example/parent'
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)

from PIL import Image

sys.path.append("..")

import model.resnet34 as resnet34
import model.cc_resnet as cc

classes = ['GENDER', 'CHF', 'PVD', 'CEVD', 'DIAB', 'DIABWC', 'AGE', 'RAF']

# Model
## Fast-AI ResNet34
# model = fastai_resnet.resnet34(num_classes=num_classes)
## PyTorch ResNet34
# model = resnet34.ResNet34(num_classes=num_classes)
## Coord Conv Model

device = torch.device("cpu")
model = resnet34.ResNet34(num_classes=8)
for name, module in model.named_modules():
	print(name, end=', ')
print('\n')

model_pth_path='hcc-cci-l1-model_best.pth'
ckpt = torch.load(model_pth_path, map_location='cpu')
if 'state_dict' in list(ckpt.keys()):
	ckpt = ckpt['state_dict']

state_dict = {}
for k, v in ckpt.items():
	if 'module.' in k:
		state_dict[k[7:]] = v
	else:
		state_dict[k] = v
model.load_state_dict(state_dict)
model.to(device)
model.eval()

transform = transforms.Compose([ transforms.Resize(256),transforms.ToTensor()])
transform_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

for root, dirnames, filenames in os.walk(os.getcwd()):
	for filename in filenames:
		if filename.endswith(".png"):
			try:
				fn=os.path.join(root, filename)
				print(fn)
				img = Image.open(fn)
				img = img.convert('RGB')
				transformed_img = transform(img)
				input = transform_normalize(transformed_img)
				input = input.unsqueeze(0)
				output = model(input)
				default_cmap = LinearSegmentedColormap.from_list('custom blue', [(0, '#ffffff'),(0.25, '#000000'), (1, '#000000')], N=256)
				torch.manual_seed(0)
				np.random.seed(0)
				occlusion = Occlusion(model)
				for i in range(len(classes)):
					print(classes[i])		
					attributions_occ = occlusion.attribute(input,strides = (3, 8, 8), target=i, sliding_window_shapes=(3,15,15),baselines=0)
					fig_tup= viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)), np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),["original_image", "heat_map"],["all", "positive"],show_colorbar=True, outlier_perc=2,)
					fig1, axis = fig_tup
					fig1.savefig(fn+classes[i]+'.png')
			except:
				fn=os.path.join(root, filename)
				print("ERROR")
				print(fn)
				print("ERROR")
				break