import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm

from dataset import ClassifierDataset
from test_dataset import TestDataset
from model.resnet34 import ResNet34
from model.cc_resnet import resnet34
from utils.metric import auroc_score
from utils.checkpoints import load_checkpoint

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# conditions = ['GENDER', 'HCC18', 'HCC22', 'HCC40', 'HCC48', 'HCC59', 'HCC85', 'HCC96', 'HCC108', 'HCC111', 'HCC138', 'AGE', 'RAF']
conditions = ['GENDER', 'HCC18', 'HCC22', 'HCC85', 'HCC96', 'HCC108', 'HCC111', 'AGE', 'RAF']
num_classes = len(conditions) - 2

def arg_parse():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default='data', help='Directory with data')
    parser.add_argument('--checkpoint_path', default='checkpoints', help='Checkpoint path')
    parser.add_argument('--out_path', default='output', help='Prediction output path')
    parser.add_argument('--size', default=256, type=int, help='Size of CT to generate')
    parser.add_argument('--age_norm', default=100.0, type=float, help='Normalization of age')
    parser.add_argument('--raf_norm', default=10.0, type=float, help='Normalization of RAF')
    parser.add_argument('--only_pred', default=False, action='store_true', help='Only generate predictions or get score')
    parser.add_argument('--calc_stats', default=False, action='store_true', help='Calculate specificity and sensitivity')
    args = parser.parse_args()
    return args

def get_prediction(code, label, thresh=0.5):
    if label == 'GENDER':
        if code <= thresh: return 'female'
        else: return 'male'
    elif 'HCC' in label:
        if code <= thresh: return 'ABSENT'
        else: return 'PRESENT'

# def calculate_accuracy(y_pred, y_label):
#     pred_conditions = y_pred[:,:-2]
#     label_conditions = y_label[:,:-2].long()
#     batch, num_classes = label_conditions.shape

#     # row 0 - false positive
#     # row 1 - false negative
#     # row 2 - true positive
#     # row 3 - true negative
#     stat = torch.zeros((4, 11))
#     correct = torch.zeros(num_classes)
#     for i in range(num_classes):
#         start = i * 3
#         end = (i + 1) * 3
#         pred = torch.max(F.softmax(pred_conditions[:, start:end], dim=-1), dim=-1)[1]
#         correct[i] = (pred == label_conditions[:, i]).sum()
#         if (pred == label_conditions).sum():
#             if pred == 0:
#                 stat[3, i] += 1
#             else:
#                 stat[2, i] += 1
#         else:
#             if pred == 0:
#                 stat[1, i] += 1
#             else:
#                 stat[0, i] += 1

#     return stat, correct

def calculate_accuracy(y_pred, y_label):
    pred_conditions = y_pred[:,:-2]
    label_conditions = y_label[:,:-2].long()
    pred_age = y_pred[:,-2]
    label_age = y_label[:,-2]
    pred_raf = y_pred[:,-1]
    label_raf = y_label[:,-1]
    batch, num_classes = label_conditions.shape

    # row 0 - false positive
    # row 1 - false negative
    # row 2 - true positive
    # row 3 - true negative
    stat = torch.zeros((4, num_classes))
    correct = torch.zeros(num_classes)
    for i in range(num_classes):
        start = i * 3
        end = (i + 1) * 3
        pred = torch.max(F.softmax(pred_conditions[:, start:end], dim=-1), dim=-1)[1]
        correct[i] = (pred == label_conditions[:, i]).sum()
        if correct[i]:
            if pred == 0:
                stat[3, i] += 1
            else:
                stat[2, i] += 1
        else:
            if pred == 0:
                stat[1, i] += 1
            else:
                stat[0, i] += 1

    return stat, correct

def test(model, test_dataloader, only_pred, stat):
    result = {}
    result['STUDY'] = []
    for c in conditions:
        result[c] = []
    tot_score = 0
    class_score = torch.zeros(num_classes)

    stats = torch.zeros((4, num_classes))
    correct = torch.zeros(num_classes)
    for i, val in tqdm(enumerate(test_dataloader)):
        if only_pred:
            img = val
        else:
            img, labels = val
            labels = labels.to(device)
        img = img.to(device)
        img = img.repeat(1, 3, 1, 1)
        result['STUDY'].append(test_dataloader.dataset.at(i))

        prediction = torch.sigmoid(model(img))
        if stat:
            x = calculate_accuracy(prediction, labels)
            stats += x[0]
            correct += x[1]
        if not only_pred:
            s, cs = auroc_score(prediction[:, :-2], labels[:, :-2])
            tot_score += s
            class_score += cs
        result['AGE'].append(prediction[:, -2].item() * args.age_norm)
        result['RAF'].append(prediction[:, -1].item() * args.raf_norm)
        #result['GENDER'].append(get_prediction(prediction[:, 0], 'GENDER'))
        result['GENDER'].append(prediction[:, 0].item())

        for j in range(1, num_classes):
            #result[conditions[j]].append(get_prediction(prediction[:, j], conditions[j]))
            result[conditions[j]].append(prediction[:, j].item())
        del val, prediction

    return result, 100 * tot_score/(i + 1), 100 * class_score/(i + 1), stats, correct/len(test_dataloader.dataset)

args = arg_parse()

print('Init model')
#model = ResNet34(num_classes=num_classes+2)
model = resnet34(pretrained=False, num_classes=num_classes+2).to(device)
load_checkpoint(args.checkpoint_path, model)
model.to(device)
#model_state = {'state_dict' : model.state_dict()}
#torch.save(model_state, 'checkpoint.pth')
model.eval()

print('Initializing')
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.55001191,), (0.18854326,))])
if args.only_pred:
    print('not using csv')
    test_dataset = TestDataset(args.data_dir, transforms=transform, size=args.size)
else:
    print('using csv')
    test_dataset = ClassifierDataset(args.data_dir, conditions[1:-2], transforms=transform, size=args.size, train=False, age_norm=args.age_norm, raf_norm=args.raf_norm)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
if not os.path.exists(args.out_path):
    os.makedirs(args.out_path)

print('Testing')
results, score, class_score, stats, correct = test(model, test_dataloader, args.only_pred, args.calc_stats)
if not args.only_pred:
    print('AUROC Score: {:.4f}'.format(score))
    for i in range(num_classes):
        print('{}: {:.4f}\t{:.4f}'.format(conditions[i], class_score[i], correct[i]*100))
if args.calc_stats:
    print('Stats [sensitivity, specificity]')
    # sensitivity = tp/(tp + fn)
    # specificity = tn/(tn + fp)
    for i in range(num_classes):
        sensitivity = stats[2, i]/(stats[2, i] + stats[1, i])
        specificity = stats[3, i]/(stats[3, i] + stats[0, i])
        print('{}:\t{:.4f}\t{:.4f}'.format(conditions[i], sensitivity, specificity))
    print(stats)

df = pd.DataFrame(results, columns=['STUDY'] + conditions)
df.to_csv(os.path.join(args.out_path, 'out.csv'), index=False)
