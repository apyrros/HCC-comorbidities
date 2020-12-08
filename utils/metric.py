import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, roc_auc_score

def auroc_score(y_preds, y_labels):
    """
    Calculate the area under the receiver operator curve for multi-class, multi-label classification
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

    @param y_pred  : scores output from classifier (not probabilities) - shape(-1, num_classes * 3),
                     3 comes from number of labels in `labels.py`.
    @param y_labels: labels for each class - shape(-1, num_classes)
    @return score : area under the curve
    """
    y_preds = y_preds.cpu().detach()
    y_labels = y_labels.cpu()
    _, num_classes = y_labels.shape

    score = 0.0
    scores = torch.zeros(num_classes)
    for i in range(num_classes):
        # extract different 3 labels for each class and obtain
        # probabilities for each label
        # scores[i] = roc_auc_score(y_labels[:, i], y_preds[:, i], average='micro')
        fpr, tpr, _ = roc_curve(y_labels[:, i], y_preds[:, i], pos_label=1)
        scores[i] = auc(fpr, tpr)
        score += scores[i]

    return score/num_classes, scores

def roc_plot(y_preds, y_labels):
    """
    Plot the receiver operator curves for multi-class, multi-label classification
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    @param y_pred  : scores output from classifier (not probabilities) - shape(-1, num_classes * 4),
                     4 comes from number of labels in `labels.py`.
    @param y_labels: labels for each class - shape(-1, num_classes)
    """
    _, num_classes = y_labels.shape

    score = 0.0
    plt.figure(figsize=(5,5))
    for i in range(num_classes):
        # extract different 4 labels for each class and obtain
        # probabilities for each label
        y_prob = F.softmax(y_preds[:, i * 2: (i + 1) * 2], dim=-1)
        y_pred = torch.argmax(y_prob, dim=-1)
        y_score = torch.zeros_like(y_prob).long()
        y_score[torch.arange(y_prob.size(0)), y_pred] = 1
        y_label = torch.zeros_like(y_prob).long()
        y_label[torch.arange(y_prob.size(0)), y_labels[:,i]] = 1
        fpr, tpr, _ = roc_curve(y_label.view(-1), y_score.view(-1))
        plt.plot(fpr, tpr, label=f'ROC curve of class {i}')

    plt.title('Receiver Operating Characteristic Curves')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    return

if __name__=="__main__":
    predictions = torch.tensor([[ 4.6874e-01,  8.2690e-01,  3.0458e-01, -9.6406e-01,  4.0846e-01,
         -1.6326e-01,  6.7063e-02, -3.5437e-01, -4.1962e-01, -4.4155e-01,
         -7.7163e-01, -5.0865e-01, -4.9974e-01, -1.2018e-01,  3.8633e-01,
          1.3763e-01, -1.5074e-01,  1.9078e-01,  2.2850e-01, -2.1904e-01,
          2.5684e-01, -5.2830e-01,  8.7226e-03, -1.1783e-02, -1.1234e+00,
          5.4753e-01, -5.8572e-01,  1.4012e+00,  4.2121e-01, -7.7597e-01,
         -1.0470e+00, -1.7869e-01,  3.8270e-01, -1.0099e-01, -3.8127e-02,
         -2.7691e-01,  7.0123e-01,  1.1265e+00, -3.2833e-01, -1.7692e-01,
          3.0048e-01, -6.7690e-01, -1.3011e-01,  2.9113e-01,  9.3907e-01,
         -6.2780e-02,  1.4889e-01,  4.9593e-01,  4.8354e-01,  6.2854e-01,
          8.0164e-01, -1.8209e-01,  1.8023e-01, -2.5064e-01, -4.2866e-01,
          1.0550e+00],
        [ 6.8068e-01,  1.2712e+00,  5.3864e-01, -6.8540e-01,  2.4200e-01,
         -2.7153e-01,  4.1159e-01, -3.5505e-01, -5.0452e-01, -8.0282e-01,
         -4.3048e-01, -3.3836e-01,  5.9083e-02,  1.3771e-01, -1.2423e-01,
          6.9446e-02,  3.2578e-01,  8.0908e-01, -3.3655e-01, -6.8823e-01,
         -4.8987e-01, -3.5055e-01,  7.5016e-01, -5.2437e-02, -9.6717e-02,
          4.1100e-01, -3.2309e-01,  1.2814e+00,  5.7204e-01, -5.7558e-01,
         -9.9394e-01,  1.0183e-01,  1.5324e-01, -5.0336e-01,  1.0690e-01,
         -6.1239e-01, -1.0496e-01,  1.5686e+00, -2.6410e-01, -6.7596e-01,
         -1.0566e-01, -3.8711e-01, -4.4683e-01,  4.6811e-01,  3.8097e-01,
         -1.8445e-01, -5.5679e-01, -9.7067e-03,  2.9416e-01,  6.4679e-01,
          6.4341e-01,  3.0528e-01,  3.9615e-02, -3.9745e-01, -8.8605e-02,
         -2.5231e-01],
        [ 9.4944e-01,  9.0502e-01,  1.7748e-01, -5.9598e-01, -5.5822e-01,
          1.7449e-01,  7.6185e-01, -8.6316e-02,  3.4435e-02, -1.0256e-01,
         -3.6371e-01,  3.1440e-01, -7.4135e-01,  3.5920e-01, -2.2656e-01,
          3.8602e-01,  1.6579e-01,  2.4694e-01,  2.2226e-01, -7.1503e-02,
          1.4764e-01, -3.9398e-01,  4.3873e-01,  2.5599e-01, -6.6837e-01,
          6.9129e-01, -5.2824e-01,  9.9403e-01, -3.9699e-02, -7.9194e-01,
         -8.1260e-01,  1.8499e-01, -1.8044e-01, -2.1213e-01,  1.0810e-01,
          3.1199e-03,  2.3603e-01,  8.8588e-01,  1.5650e-01, -2.9020e-02,
         -3.8045e-01, -3.4449e-01,  1.2009e-01, -2.9346e-01,  1.0959e+00,
          4.0973e-01, -1.9673e-01,  2.3521e-01, -5.3992e-02,  9.7810e-01,
          7.3937e-01,  8.5327e-01,  9.7428e-02, -4.0751e-01, -3.2482e-01,
          9.6672e-02],
        [ 7.0772e-01,  1.8961e+00,  1.2815e-02, -9.1997e-01, -1.1545e-01,
          1.9465e-01,  5.1845e-01,  2.1188e-01, -7.0486e-01, -1.0168e+00,
         -2.2078e-01,  2.9346e-01, -5.6392e-01, -2.5380e-01,  6.9668e-01,
          4.5639e-01,  1.7345e-01,  5.4072e-01,  2.5446e-01, -6.5915e-02,
         -5.4800e-01,  1.2858e-02, -5.1783e-02, -2.3539e-01, -4.6099e-01,
          9.6994e-01,  1.5139e-01,  1.0489e+00, -4.3214e-01, -6.8232e-01,
         -1.1121e+00,  2.3392e-01, -2.2636e-01, -9.8626e-01,  1.7549e-01,
         -1.4973e-01, -8.9324e-04,  1.2902e+00,  2.9013e-01, -1.2420e-02,
          2.8163e-01, -8.3933e-02, -4.3580e-01,  2.6033e-01,  9.2898e-01,
          5.9948e-01, -1.8716e-01, -4.4953e-01,  4.2350e-02,  7.2141e-01,
          4.0597e-01,  8.3610e-01, -3.5355e-01, -5.9831e-01, -4.8664e-01,
         -1.1491e-01]])
    labels = torch.tensor([[3, 3, 3, 0, 3, 3, 3, 3, 3, 2, 0, 3, 3, 1],
        [3, 3, 0, 0, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3],
        [3, 3, 0, 0, 3, 0, 3, 3, 3, 3, 0, 3, 3, 0],
        [3, 3, 3, 3, 3, 3, 3, 3, 0, 1, 3, 3, 3, 0]])

    num_classes = labels.size(-1)
    print(auroc_score(predictions, labels) / num_classes)
    roc_plot(predictions, labels)

