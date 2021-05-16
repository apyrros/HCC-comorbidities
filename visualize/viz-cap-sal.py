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
from captum.attr import Saliency
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from captum.attr import DeepLift

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

classes = ['GENDER', 'HCC18', 'HCC22', 'HCC85', 'HCC96', 'HCC108', 'HCC111', 'AGE', 'RAF']

def attribute_image_features(algorithm, input, **kwargs):
    model.zero_grad()
    tensor_attributions = algorithm.attribute(input,target=1,**kwargs)
    return tensor_attributions


device = torch.device("cpu")
model = cc.resnet34(pretrained=False, num_classes=9).to(device)
for name, module in model.named_modules():
	print(name, end=', ')
print('\n')

model_pth_path='HCC-bce-cordconv-model_best.pth'
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
			#try:
			fn=os.path.join(root, filename)
			print(fn)
			img = Image.open(fn)
			img = img.convert('RGB')
			transformed_img = transform(img)
			input = transform_normalize(transformed_img)
			input = input.unsqueeze(0)
			saliency = Saliency(model)
			default_cmap = LinearSegmentedColormap.from_list('custom blue', [(0, '#ffffff'),(0.25, '#000000'), (1, '#000000')], N=256)
			torch.manual_seed(0)
			np.random.seed(0)
			for i in range(len(classes)):
				print(classes[i])
				original_image = np.transpose((transformed_img.cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))
				grads = saliency.attribute(input, target=i)
				grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))
				fig_tup = viz.visualize_image_attr(grads, original_image, method="blended_heat_map", sign="absolute_value", show_colorbar=True, title="Overlayed Gradient Magnitudes")
				fig1, axis = fig_tup
				fig1.savefig(fn+classes[i]+'.png')
			#except:
			#	fn=os.path.join(root, filename)
			#	print("ERROR")
			#	print(fn)
			#	print("ERROR")
			#	break