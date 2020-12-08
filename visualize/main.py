#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-18

from __future__ import print_function

import copy
import os
import os.path as osp
import sys

import click
import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms

from grad_cam import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)

sys.path.append("..")
import model.fastai_resnet as fastai_resnet
import model.resnet34 as resnet34
import model.cc_resnet as cc

# if a model includes LSTM, such as in image captioning,
# torch.backends.cudnn.enabled = False


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def load_images(image_folder, image_paths):
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        if '.png' not in image_path:
            continue
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(os.path.join(image_folder, image_path))
        images.append(image)
        raw_images.append(raw_image)
    return images, raw_images


def load_image_names(image_path):
    images = [image_path.rstrip('.png') for image_path in image_path if '.png' in image_path]
    return images


def get_classtable_fastai():
    labels = ['GENDER-NA', 'GENDER-FEMALE', 'GENDER-MALE']
    conditions = ['HCC18', 'HCC22', 'HCC40', 'HCC48', 'HCC59', 'HCC85', 'HCC96', 'HCC108', 'HCC111', 'HCC138']
    for condition in conditions:
        labels.append(condition + '-ABSENT')
        labels.append(condition + '-NA')
        labels.append(condition + '-PRESENT')
    labels.append('RAF')
    labels.append('AGE')
    return labels


def get_classtable():
    labels = ['GENDER-FEMALE', 'GENDER-MALE', 'GENDER-NA']
    conditions = ['HCC18', 'HCC22', 'HCC40', 'HCC48', 'HCC59', 'HCC85', 'HCC96', 'HCC108', 'HCC111', 'HCC138']
    for condition in conditions:
        labels.append(condition + '-ABSENT')
        labels.append(condition + '-PRESENT')
        labels.append(condition + '-NA')
    labels.append('RAF')
    labels.append('AGE')
    return labels


def preprocess(image_path):
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (256,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.459882, 0.459882, 0.459882], std=[0.220424, 0.220424, 0.220424]),
            transforms.Normalize((0.55001191,0.55001191,0.55001191), (0.18854326,0.18854326,0.18854326)),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image


def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))


def save_sensitivity(filename, maps):
    maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (256, 256), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename, maps)

def save_output(filename, scores, classes):
    with open(filename, 'w') as fptr:
        fptr.write('Model predictions for sample {}.\n'.format(filename.split("/")[-2]))
        fptr.write('===============================================================\n')
        for i in range(11):
            s = i * 3
            e = (i + 1) * 3
            p = np.argmax(scores[s:e].detach().numpy())
            c = classes[s].split('-')[0]
            l = "{}: {}\n".format(c, classes[s + p].split('-')[1])
            fptr.write(l)
            l = "Scores: "
            for j in range(3):
                l += "{} - {:.2f}%, ".format(classes[s + j], 100 * scores[s + j])

            fptr.write(l + '\n')
            fptr.write('===============================================================\n')

        fptr.write("{}: {}\n".format(classes[33], 10 * scores[33]))
        fptr.write('===============================================================\n')
        fptr.write("{}: {}\n".format(classes[34], 100 * scores[34]))
        fptr.write('===============================================================\n')


def save_output_bce(filename, scores, classes):
    with open(filename, 'w') as fptr:
        fptr.write('Model predictions for sample {}.\n'.format(filename.split("/")[-2]))
        fptr.write('===============================================================\n')
        for i in range(len(classes)):
            if i == len(classes) - 2:
                scores[i] *= 100
            elif i == len(classes) - 1:
                scores[i] *= 10
            fptr.write("{}: {}\n".format(classes[i], scores[i]))
            fptr.write('===============================================================\n')

# torchvision models
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


@click.group()
@click.pass_context
def main(ctx):
    print("Mode:", ctx.invoked_subcommand)

@main.command()
@click.option("-m", "--model-pth-path", type=str, required=True)
@click.option("-i", "--image-dir", type=str, required=True)
@click.option("-c", "--num_classes", type=int, default=35)
@click.option("-o", "--output-dir", type=str, default="./results")
@click.option("-s", "--start-class", type=int, default=0)
@click.option("-b", "--batch-size", type=int, default=-1)
@click.option("--cuda/--cpu", default=False)
def test(model_pth_path, image_dir, num_classes, output_dir, start_class, batch_size, cuda):
    """
    Generate Grad-CAM at different layers of FastAi ResNet-34
    """
    device = get_device(cuda)

    # Synset words TODO: change back
    ## classes for model Ayis trained with fastai
    # classes = get_classtable_fastai()
    ## classes for cross entropy training
    # classes = get_classtable()
    ## classes for BCE training
    classes = ['GENDER', 'HCC18', 'HCC22', 'HCC85', 'HCC96', 'HCC108', 'HCC111', 'AGE', 'RAF']


    # Model
    ## Fast-AI ResNet34
    # model = fastai_resnet.resnet34(num_classes=num_classes)
    ## PyTorch ResNet34
    # model = resnet34.ResNet34(num_classes=num_classes)
    ## Coord Conv Model
    model = cc.resnet34(pretrained=False, num_classes=9).to(device)
    ## print out names of all layers
    for name, module in model.named_modules():
        print(name, end=', ')
    print('\n')
    # exit()

    ckpt = torch.load(model_pth_path, map_location='cpu')
    if 'state_dict' in list(ckpt.keys()):
        ckpt = ckpt['state_dict']

    state_dict = {}
    for k, v in ckpt.items():
        if 'module.' in k:
            state_dict[k[7:]] = v
        else:
            state_dict[k] = v

    # print(state_dict.keys())
    # exit()

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # The four residual layers
    # target_layers = ["conv1", "layer1.2.conv2", "layer2.3.conv2", "layer3.5.conv2", "layer4.2.conv2", "head.layers.5"]
    target_layers = ["conv1", "layer1.2", "layer2.3", "layer3.5", "layer4.2", "head.layers"]
    # layer_names = ["conv1", "layer1", "layer2", "layer3", "layer4", "fc"]
    layer_names = ["conv1", "layer1", "layer2", "layer3", "layer4", "head"]
    # target_layers = ["conv1", "layer1.2.conv2", "layer2.3.conv2", "layer3.5.conv2", "layer4.2.conv2", "fc"]
    # for i in range(len(target_layers)):
    #     target_layers[i] = 'resnet34.' + target_layers[i]

    paths = [image_path for image_path in os.listdir(image_dir) if '.png' in image_path]
    if batch_size == -1:
        batch_size = len(paths)
    for b in range(int(np.ceil(len(paths) / batch_size))):
        # Images
        image_paths = paths[b * batch_size: min((b + 1) * batch_size, len(paths))]
        images, raw_images = load_images(image_dir, image_paths)
        images = torch.stack(images).to(device)
        image_names = load_image_names(image_paths)

        for i in range(start_class, num_classes):
            print(classes[i], i, num_classes)
            target_class = i
            gcam = GradCAM(model=model)
            outputs, scores = gcam.forward(images)
            ids = torch.stack(images.size(0) * [torch.arange(outputs.size(-1))])
            ids_ = torch.LongTensor([[target_class]] * len(images)).to(device)
            gcam.backward(ids=ids_)

            # write model outputs to files
            if i == 0:
                for j in range(len(image_names)):
                    savedir = os.path.join(output_dir, image_names[j])
                    if not os.path.exists(savedir):
                        os.makedirs(savedir)
                    save_output_bce(os.path.join(savedir, 'outputs.txt'), scores[j], classes)

            for target_layer, layer_name in zip(target_layers, layer_names):
                print("Generating Grad-CAM @{} for class {}.".format(target_layer, classes[target_class]))

                # Grad-CAM
                regions = gcam.generate(target_layer=target_layer)

                for j in range(len(images)):
                    savedir = os.path.join(output_dir, image_names[j], 'gradcam', layer_name)
                    if not os.path.exists(savedir):
                        os.makedirs(savedir)

                    save_gradcam(
                        filename=osp.join(
                            savedir,
                            "{}.png".format(classes[target_class]),
                        ),
                        gcam=regions[j, 0],
                        raw_image=raw_images[j],
                    )


if __name__ == "__main__":
    main()
