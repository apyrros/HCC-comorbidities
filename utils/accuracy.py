import torch

class Accuracy():
    def __init__(self, num_classes, num_entries):
        self.num_classes = num_classes
        self.correct = torch.empty(self.num_classes)
        self.num_entries = num_entries

    def calculate_accuracy(y_pred, y_label):
        pred_conditions = y_pred[:,:-2]
        label_conditions = y_label[:,:-2]

        for i in range(self.num_classes):
            start = i * 3
            end = (i + 1) * 3
            pred = torch.max(F.softmax(pred_conditions[:, start:end], dim=-1), dim=-1)[1]
            self.correct[i] = (pred == label_conditions[:, i]).sum()

        return self.correct

    def print_accuracy():
        for i, acc in enumerate(self.correct/):
            print('\t{}: {:.4f}%'.format(conditions[i], acc*100))
