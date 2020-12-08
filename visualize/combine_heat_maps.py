import os
import numpy as np

from PIL import Image
from argparse import ArgumentParser


parser = ArgumentParser(description="Arguments for training")
parser.add_argument('data', help="Path to where data is stored.")
args =  parser.parse_args()

for i, study in enumerate(os.listdir(args.data)):
    inner_path = os.path.join(args.data, study, 'gradcam')
    if not os.path.exists(inner_path):
        continue

    # layers = ["conv1", "layer1.2.conv2", "layer2.3.conv2", "layer3.5.conv2", "layer4.2.conv2", "head.layers.5"]
    layers = ["conv1", "layer1", "layer2", "layer3", "layer4", "head"]
    #files = ['HCC48-NA.png', 'HCC85-PRESENT.png', 'HCC138-PRESENT.png', 'HCC108-NA.png', 'HCC18-PRESENT.png',
    #         'HCC96-PRESENT.png', 'HCC59-PRESENT.png', 'GENDER-NA.png', 'HCC85-NA.png', 'HCC40-PRESENT.png',
    #         'HCC40-NA.png', 'RAF.png', 'HCC138-NA.png', 'HCC22-PRESENT.png', 'GENDER-FEMALE.png',
    #         'HCC138-ABSENT.png', 'HCC96-ABSENT.png', 'HCC59-NA.png', 'HCC59-ABSENT.png', 'HCC108-ABSENT.png',
    #         'HCC108-PRESENT.png', 'HCC48-PRESENT.png', 'HCC48-ABSENT.png', 'HCC22-NA.png', 'HCC18-ABSENT.png',
    #         'HCC111-PRESENT.png', 'HCC18-NA.png', 'HCC85-ABSENT.png', 'HCC111-NA.png', 'GENDER-MALE.png',
    #         'HCC111-ABSENT.png', 'HCC40-ABSENT.png', 'HCC96-NA.png', 'AGE.png', 'HCC22-ABSENT.png']
    files = ['AGE.png','GENDER.png','HCC108.png','HCC111.png','HCC18.png','HCC22.png','HCC85.png','HCC96.png','RAF.png']

    # make max dir
    max_path = os.path.join(inner_path, 'max_over_layers')
    if not os.path.exists(max_path):
        os.makedirs(max_path)

    # make avg dir
    avg_path = os.path.join(inner_path, 'avg_over_layers')
    if not os.path.exists(avg_path):
        os.makedirs(avg_path)

    for file in files:
        image_layers = []
        for layer in layers:
            if 'head' in layer:
                continue
            image_layers.append(np.asarray(Image.open(os.path.join(inner_path, layer, file))))

        image_layers = np.asarray(image_layers)

        avg_layers = Image.fromarray(np.mean(image_layers, axis=0).astype(np.uint8))
        avg_layers.save(os.path.join(avg_path, file))

        max_layers = Image.fromarray(np.max(image_layers, axis=0).astype(np.uint8))
        max_layers.save(os.path.join(max_path, file))

