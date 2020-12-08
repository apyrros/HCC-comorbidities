import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn import calibration
from argparse import ArgumentParser

parser = ArgumentParser(description='Generate ROC, PR, and Calibration Curves given predicted scores and true labels')
parser.add_argument('--predictions', '-p', default='out.csv', help='CSV containing all of the probability scores')
parser.add_argument('--labels', '-l', default='new_test.csv', help='CSV containing all of the true labels')
parser.add_argument('--output_dir', '-o', default='outputs/', help='Directory in which to store all the plots')
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

labels = pd.read_csv(args.labels)
predicts = pd.read_csv(args.predictions)

labels = labels.drop(columns=['FILE', 'AGE', 'RAF', 'HCC40', 'HCC48', 'HCC59', 'HCC138', 'GENDER'])
# predicts = predicts.drop(columns=['STUDY', 'AGE', 'RAF', 'HCC40', 'HCC48', 'HCC59', 'HCC138', 'GENDER'])
predicts = predicts.drop(columns=['STUDY', 'AGE', 'RAF', 'GENDER'])

conditions = list(predicts.columns)
print(conditions)

def convert_labels(df):
    # convert df to proper labels
    ## skipping GENDER because predictions are almost perfect
    # df.loc[(df.GENDER == 'female'),'GENDER'] = 0
    # df.loc[(df.GENDER == 'male'),'GENDER'] = 1

    # go through all hcc codes and convert to labels
    for c in conditions:
        df.loc[(df[c] == 'ABSENT'), c] = 0
        df.loc[(df[c] == 'PRESENT'), c] = 1

convert_labels(labels)

# getting ROC data
roc_dict = {}

for condition in conditions:
    y = np.asarray(labels[condition], dtype=np.int32)
    score = np.asarray(predicts[condition], dtype=np.float32)
    fpr, tpr, thresh = metrics.roc_curve(y, score, pos_label=1)
    area = metrics.auc(fpr, tpr)
    roc_dict[condition] = (fpr, tpr, thresh, area)

y = np.asarray(labels, dtype=np.int32).flatten()
score = np.asarray(predicts, dtype=np.float32).flatten()
fpr, tpr, thresh = metrics.roc_curve(y, score, pos_label=1)
area = metrics.auc(fpr, tpr)
roc_dict['TOTAL'] = (fpr, tpr, thresh, area)

# plot ROC figure

plt.figure(figsize=(10, 10))
plt.plot([0, 1], [0, 1], color='black', linestyle='dashed')

for key in roc_dict.keys():
    fpr, tpr, _, area = roc_dict[key]
    plt.plot(fpr, tpr, label=key + ' - AUC : {:.4f}'.format(area))

plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc=4)
plt.xlim(-0.01, 1.01)
plt.ylim(-0.01, 1.01)
plt.savefig(os.path.join(args.output_dir, 'roc.png'), bbox_inches='tight')
# plt.show()

# get PR data
pr_dict = {}

for condition in conditions:
    y = np.asarray(labels[condition], dtype=np.int32)
    score = np.asarray(predicts[condition], dtype=np.float32)
    precision, recall, thresh = metrics.precision_recall_curve(y, score, pos_label=1)
    ap = metrics.average_precision_score(y, score)
    pr_dict[condition] = (precision, recall, thresh, ap)

y = np.asarray(labels, dtype=np.int32).flatten()
score = np.asarray(predicts, dtype=np.float32).flatten()
precision, recall, thresh = metrics.precision_recall_curve(y, score, pos_label=1)
ap = metrics.average_precision_score(y, score)
pr_dict['TOTAL'] = (precision, recall, thresh, ap)

# plot PR figure

plt.figure(figsize=(10, 10))

for key in roc_dict.keys():
    p, r, _, ap = pr_dict[key]
    plt.plot(r, p, label=key + ' - AP : {:.4f}'.format(ap))

plt.ylabel('Precision')
plt.xlabel('Recall')
plt.title('Precision-Recall Curve')
plt.legend(loc=3)
plt.xlim(-0.01, 1.01)
plt.ylim(-0.01, 1.01)
plt.savefig(os.path.join(args.output_dir, 'pr.png'), bbox_inches='tight')
# plt.show()

# get Calibration data
cal_dict = {}

for condition in conditions:
    y = np.asarray(labels[condition], dtype=np.int32)
    score = np.asarray(predicts[condition], dtype=np.float32)
    frac_pos, mean_pred = calibration.calibration_curve(y, score, n_bins=10)
    brier = metrics.brier_score_loss(y, score, pos_label=1)
    cal_dict[condition] = (frac_pos, mean_pred, brier)

y = np.asarray(labels, dtype=np.int32).flatten()
score = np.asarray(predicts, dtype=np.float32).flatten()
frac_pos, mean_pred = calibration.calibration_curve(y, score, n_bins=10)
brier = metrics.brier_score_loss(y, score, pos_label=1)
cal_dict['TOTAL'] = (frac_pos, mean_pred, brier)

# plot Calibration figure

plt.figure(figsize=(10, 10))
plt.plot([0, 1], [0, 1], color='black', linestyle='dashed')

for key in roc_dict.keys():
    frac_pos, mean_pred, brier = cal_dict[key]
    plt.plot(mean_pred, frac_pos, 's-', label=key + ' - Brier : {:.4f}'.format(brier))

plt.ylabel('Fraction of Positives')
plt.xlabel('Mean Predicted Value')
plt.title('Calibration Plots (Reliability Curves)')
plt.legend(loc=4)
plt.xlim(-0.01, 1.01)
plt.ylim(-0.01, 1.01)
plt.savefig(os.path.join(args.output_dir, 'cal.png'), bbox_inches='tight')
# plt.show()
