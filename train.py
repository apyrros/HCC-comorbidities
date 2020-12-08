import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsampler import ImbalancedDatasetSampler
from time import time
from argparse import ArgumentParser

from dataset import ClassifierDataset
from model.resnet34 import ResNet34
#from model.fastai_resnet import resnet34
#from model.coord_conv_resnet import resnet34
from model.cc_resnet import resnet34
from utils.loss import *
from utils.checkpoints import *
from utils.metric import auroc_score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# conditions = ['gender', 'HCC18', 'HCC22', 'HCC40', 'HCC48', 'HCC59', 'HCC85', 'HCC96', 'HCC108', 'HCC111', 'HCC138', 'age MSE', 'raf MSE']
conditions = ['gender', 'HCC18', 'HCC22', 'HCC85', 'HCC96', 'HCC108', 'HCC111', 'age MSE', 'raf MSE']
num_classes = len(conditions) - 2

def arg_parse():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default='data', help='Directory with data')
    parser.add_argument('--checkpoint_dir', default='checkpoints', help='Checkpoint output directory')
    parser.add_argument('--pretrain', default=None, help='Path to pretrained weights')
    parser.add_argument('--size', default=256, type=int, help="Size of CT to generate")
    parser.add_argument('--age_norm', default=100.0, type=float, help="Normalization of age")
    parser.add_argument('--raf_norm', default=10.0, type=float, help="Normalization of RAF")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate")
    parser.add_argument('--epochs', default=1000, type=int, help="Number of epochs")
    parser.add_argument('--train_batch_size', default=64, type=int, help="Training batch size")
    parser.add_argument('--test_batch_size', default=1, type=int, help="Testing batch size")
    parser.add_argument('--num_workers', default=32, type=int, help="Number of workers used")
    parser.add_argument('--decay_start_epoch', default=15, type=int, help="Epoch to start learning rate decay")
    args = parser.parse_args()
    return args

def calculate_accuracy(y_pred, y_label, threshold=0.5):
    pred_conditions = y_pred[:,:-2]
    label_conditions = y_label[:,:-2]
    pred_age = y_pred[:,-2]
    label_age = y_label[:,-2]
    pred_raf = y_pred[:,-1]
    label_raf = y_label[:,-1]
    loss = nn.MSELoss()
    batch, _ = label_conditions.shape

    correct = torch.zeros(num_classes + 2)
    one = torch.ones(1).to(device)
    zero = torch.zeros(1).to(device)
    for i in range(num_classes):
        pred_labels = torch.where(pred_conditions[:, i] > threshold, one, zero)
        correct[i] = (pred_labels == label_conditions[:, i]).sum()

    correct[-2] = loss(pred_age, label_age)
    correct[-1] = loss(pred_raf, label_raf)

    return correct

# def print_accuracy(accuracy):
#     for i, acc in enumerate(accuracy):
#         print('{}: {:.4f}%'.format(conditions[i], acc*100))

# def print_accuracy(train_accuracy, test_accuracy):
#     for i, (train_acc, test_acc) in enumerate(zip(train_accuracy, test_accuracy)):
#         if i > 10:
#             print('{}\t: {:.4f}\t\t\t{:.4f}'.format(conditions[i], train_acc, test_acc))
#         else:
#             print('{}\t: {:.4f}%\t\t\t{:.4f}%'.format(conditions[i], train_acc*100, test_acc*100))

def print_accuracy(train_accuracy, test_accuracy, train_auroc, test_auroc):
    print('\t\tTrain accuracy:\tTest accuracy:\tTrain AUROC:\tTest AUROC:')
    for i, (train_acc, test_acc, train_au, test_au) in enumerate(zip(train_accuracy, test_accuracy, train_auroc, test_auroc)):
        print('{}\t:\t{:.4f}%\t{:.4f}%\t{:.4f}\t\t{:.4f}'.format(conditions[i], train_acc*100, test_acc*100, train_au, test_au))
    print('{}\t:\t{:.4f}\t\t{:.4f}'.format(conditions[-2], train_accuracy[-2], test_accuracy[-2]))
    print('{}\t:\t{:.4f}\t\t{:.4f}'.format(conditions[-1], train_accuracy[-1], test_accuracy[-1]))

def print_auroc(train_auroc, test_auroc):
    for i in range(num_classes):
        print('{}\t: {:.4f}\t\t{:.4f}'.format(conditions[i], train_auroc[i], test_auroc[i]))

def train_epoch(model, dataloader, optimizer):
    tot_loss = 0
    accuracy = torch.zeros(num_classes+2, dtype=torch.float32)
    tot_score = 0
    class_score = torch.zeros(num_classes)
    for i, (img, labels) in enumerate(dataloader):
        img = img.to(device)
        labels = labels.to(device)
        img = img.repeat(1, 3, 1, 1)

        prediction = torch.sigmoid(model(img))
        s, cs = auroc_score(prediction[:, :-2], labels[:, :-2])
        tot_score += s
        class_score += cs

        loss = multilabel_regression_loss(prediction, labels)
        tot_loss += loss.item()

        accuracy += calculate_accuracy(prediction, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del img, labels, prediction
    accuracy /= len(dataloader.dataset)

    return tot_loss/(i+1), accuracy, 100 * tot_score/(i + 1), 100 * class_score/(i + 1)

def train(model, train_dataloader, test_dataloader, optimizer, scheduler, epochs):
    best_loss = np.inf

    for epoch in range(epochs):
        model.train()

        start_train_time = time()
        train_loss, train_accuracy, train_auroc, train_class_auroc = train_epoch(model, train_dataloader, optimizer)
        scheduler.step(train_loss)
        train_time = time() - start_train_time

        start_test_time = time()
        test_loss, test_accuracy, test_auroc, test_class_auroc = test(model, test_dataloader)
        test_time = time() - start_test_time

        print('Epoch: [{}/{}], Train loss: {:.4f}, Test loss: {:.4f}, Train score: {:.2f}, Test score: {:.2f} Train time: {:.2f}s, Test time: {:.2f}s'.format(epoch+1, epochs, train_loss, test_loss, train_auroc, test_auroc, train_time, test_time))
        #print('\t\tTrain AUROC:\tTest AUROC:')
        #print_auroc(train_class_auroc, test_class_auroc)
        #print('\t\t  Train accuracy:\tTest accuracy:')
        print_accuracy(train_accuracy, test_accuracy, train_class_auroc, test_class_auroc)

        train_state = {'epoch'      : epoch + 1,
                       'state_dict' : model.state_dict(),
                       'optim_dict' : optimizer.state_dict()}
        model_state = {'state_dict' : model.state_dict()}
        is_best = test_loss < best_loss
        if is_best:
            best_loss = test_loss
        save_checkpoint(train_state, model_state, is_best, args.checkpoint_dir)

def test(model, test_dataloader):
    tot_loss = 0
    accuracy = torch.zeros(num_classes+2, dtype=torch.float32)
    tot_score = 0
    class_score = torch.zeros(num_classes)

    model.eval()
    with torch.no_grad():
        for i, (img, labels) in enumerate(test_dataloader):
            img = img.to(device)
            labels = labels.to(device)
            img = img.repeat(1, 3, 1, 1)

            prediction = torch.sigmoid(model(img))
            s, cs = auroc_score(prediction[:, :-2], labels[:, :-2])
            tot_score += s
            class_score += cs

            loss = multilabel_regression_loss(prediction, labels)
            tot_loss += loss.item()

            accuracy += calculate_accuracy(prediction, labels)

            del img, labels, prediction

    accuracy /= len(test_dataloader.dataset)
    return tot_loss/(i+1), accuracy, 100 * tot_score/(i + 1), 100 * class_score/(i + 1)

args = arg_parse()
print('Parameters:\n\
       data directory: {}\n\
       checkpoint directory: {}\n\
       size: {}\n\
       lr: {}\n\
       epochs: {}\n\
       train batch size: {}\n\
       number of workers: {}\n\
       epoch to start decay: {}\n'.format(args.data_dir,
       args.checkpoint_dir, args.size, args.lr, args.epochs,
       args.train_batch_size, args.num_workers,
       args.decay_start_epoch))

print('Loading data')
transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.2),
                                transforms.RandomApply([transforms.RandomAffine(10),
                                transforms.RandomResizedCrop(256, scale=(1.0, 1.1), ratio=(0.75, 1.33)),
                                transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.75),
                                transforms.RandomPerspective(distortion_scale=0.2, p=0.75),
                                transforms.ToTensor(), transforms.Normalize((0.55001191,), (0.18854326,))])
#transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.55001191,), (0.18854326,))])
# Train data: mean = 0.55001191, std = 0.18854326
train_dataset = ClassifierDataset(args.data_dir, conditions[1:-2], transforms=transform, size=args.size, train=True, age_norm=args.age_norm, raf_norm=args.raf_norm)
# Test data: mean = 0.42215754, std = 0.18558778
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.55001191,), (0.18854326,))])
test_dataset = ClassifierDataset(args.data_dir, conditions[1:-2], transforms=transform, size=args.size, train=False, age_norm=args.age_norm, raf_norm=args.raf_norm)
train_dataloader = DataLoader(dataset=train_dataset,batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)
#print('imbalanced sampler')
#train_dataloader = DataLoader(dataset=train_dataset,sampler=ImbalancedDatasetSampler(train_dataset),batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)
print('Initializing')
if args.pretrain:
    print('resnet chexpert')
    model = ResNet34(2 * 14, pretrained=True).to(device)
    #print('coord conv chexpert')
    #model = resnet34(pretrained=False, num_classes=2*14).to(device)

    load_checkpoint(args.pretrain, model)
    model.resnet34.fc = nn.Linear(model.resnet34.fc.in_features, num_classes+2)
    #head = list(model.head.layers.children())
    #in_features = 512
    #head = head[:-1]
    #model.head.layers = nn.Sequential(*head)
    #model.head.layers = nn.Sequential(model.head.layers, nn.Linear(in_features, num_classes+2))
else:
    #print('pretrained torch resnet34')
    #model = ResNet34(3*num_classes+2, pretrained=True)
    print('CoordConv resnet34 implementation')
    model = resnet34(pretrained=False, num_classes=3*num_classes+2).to(device)

if (torch.cuda.device_count() > 1):
        device_ids = list(range(torch.cuda.device_count()))
        print("GPU devices being used: ", device_ids)
        model = nn.DataParallel(model, device_ids=device_ids)

model = model.to(device)

#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
#scheduler = optim.lr_scheduler.StepLR(optimizer, args.decay_start_epoch, gamma=0.1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

print('Training')
train(model, train_dataloader, test_dataloader, optimizer, scheduler, args.epochs)
