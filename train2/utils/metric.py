from typing import Dict, Optional, Any, Callable, List
import torch
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from train.train_consts import *
from inference.post_processing.raw_pred_to_precision import ThresholdToPrecision
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt
import tempfile
from collections import defaultdict
import dill as pickle

from sklearn.metrics import precision_recall_curve


class PRAUC(torchmetrics.Metric):
    def __init__(
            self,
            compute_on_step: bool = True,
            dist_sync_on_step: bool = False,
            process_group: Optional[Any] = None,
            dist_sync_fn: Callable = None,
    ) -> None:
        super().__init__(compute_on_step, dist_sync_on_step, process_group, dist_sync_fn)
        self.pr_curve = torchmetrics.PrecisionRecallCurve(pos_label=1)
        self.auc = torchmetrics.AUC()
        self.precision = None
        self.recall = None

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.pr_curve(preds, target)

    def compute(self):
        self.precision, self.recall, _ = self.pr_curve.compute()
        self.auc(self.recall, self.precision)
        res = self.auc.compute()
        self.auc.reset()
        return res

    def reset(self):
        self.pr_curve.reset()
        self.auc.reset()


class RecallArPr(torchmetrics.Metric):
    def __init__(
            self,
            precision=0.8,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.precision = precision

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # TODO: rewrite if using batch updates
        self.pretopr = ThresholdToPrecision(target.numpy(), preds.numpy())

    def compute(self):
        return self.pretopr.map_pr(self.precision)


class Lift(torchmetrics.Metric):
    def __init__(
            self,
            percentage=0.1,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.percentage = percentage
        self.preds = []
        self.labels = []
        self.res = []

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.preds = np.concatenate([self.preds, preds.numpy()])
        self.labels = np.concatenate([self.labels, target.numpy()])

    def compute(self):
        self.sorted_labels = self.labels[np.argsort(self.preds)[::-1]]
        self.res_mean = np.mean(self.labels)

        self.res = []
        curr = 0
        for i, x in enumerate(self.sorted_labels):
            if x == 1:
                curr += 1
            self.res.append((curr / (i + 1)) / self.res_mean)

        return self.res[int(len(self.res) * self.percentage)]


def init_metrics(age_groups):
    metrics_dict = defaultdict(dict)
    for age_group in [''] + list(np.unique(age_groups)):
        for cond, cond_type in CONDITIONS.items():
            target = f'{cond}_{age_group}' if age_group else cond
            if cond_type == CLASS_TASK:
                metrics_dict["accuracy"][target] = torchmetrics.Accuracy(threshold=THRESHOLD)
                metrics_dict["auroc"][target] = torchmetrics.AUROC(num_classes=2)
                metrics_dict["prauc"][target] = PRAUC()
                metrics_dict["recall_0.8"][target] = RecallArPr()
                metrics_dict["lift_0.1"][target] = Lift()
            else:
                metrics_dict["mae"][target] = torchmetrics.MeanAbsoluteError()
    return metrics_dict


def update_metrics(y_pred: torch.Tensor, y_label: torch.Tensor, age_groups: List,
                   metrics_dict: Dict) -> dict:
    """
    Update metrics at the end of one mini-batch processing in the train loop
    :param age_groups: a list with age_group for each label
    :param y_pred: tensor with predicted values (after sigmoid for classification)
    :param y_label: tensor with ground truth labels
    :return: dict with metrics names in keys and metrics values in values
    """
    metrics_values = {}
    for metric in metrics_dict.keys():
        metrics_values[metric] = {}
        for idx, (target, cond_type) in enumerate(CONDITIONS.items()):
            for age_group in np.unique(age_groups):
                target_age = f'{target}_{age_group}'
                if target_age not in metrics_dict[metric]:
                    continue
                age_idx = np.array(age_groups) == age_group
                metrics_values[metric][target_age] = metrics_dict[metric][target_age](y_pred[age_idx, idx],
                                                    y_label[age_idx, idx].type(dtype=torch.int32) if cond_type == CLASS_TASK else y_label[age_idx, idx])
            if target not in metrics_dict[metric]:
                    continue
            metrics_values[metric][target] = metrics_dict[metric][target](y_pred[:, idx],
                                                        y_label[:, idx].type(dtype=torch.int32) if cond_type == CLASS_TASK else y_label[:, idx])
    return metrics_values


writer = SummaryWriter(f"./logs/{EXPERIMENT_NAME}")


def log_metrics(metrics_dict, age_groups: List,
                metrics_values: dict, loss: float, phase: str, epoch: int, log_plots: bool = False):
    """
    Log metrics at the end of the epoch
    :param metrics_dict: a dictionary with metrics classes
    :param age_groups: a list with all age groups
    :param log_plots: whether to log plots to mlflow
    :param metrics_values: dict with metrics names in keys and metrics values in values
    :param loss: loss value
    :param phase: "train" or "test"
    :param epoch: epoch number
    """
    writer.add_scalar(f"loss/{phase}", loss, epoch)
    for metric in metrics_values.keys():
        for target in metrics_values[metric].keys():
            writer.add_scalar(f"{metric}_{target}/{phase}", metrics_values[metric][target], epoch)
            def filter_(x):
                return ''.join([ch for ch in x.replace(',', '-') if ch.isdigit() or ch.isalpha() or ch in ['_', '-', '.', '/']])
                       
            mlflow.log_metric(filter_(f"{metric}_{target}/{phase}"), float(metrics_values[metric][target]))
            if log_plots and target != "AVG":
                if isinstance(metrics_dict[metric][target], PRAUC):
                    fig = sns.lineplot(
                        x=metrics_dict[metric][target].recall,
                        y=metrics_dict[metric][target].precision
                    ).get_figure()
                    artifact_name = f"{target}_pr_curve"
                    with tempfile.NamedTemporaryFile(suffix=".png") as f:
                        fig.savefig(f)
                        mlflow.log_artifact(local_path=f.name, artifact_path=f"{phase}/{epoch}/{artifact_name}")
                    plt.clf()
                if isinstance(metrics_dict[metric][target], Lift):
                    fig = sns.lineplot(
                        x=np.array(range(len(metrics_dict[metric][target].res))) / len(
                            metrics_dict[metric][target].res),
                        y=metrics_dict[metric][target].res,
                    ).get_figure()
                    artifact_name = f"{target}_lift_curve"
                    with tempfile.NamedTemporaryFile(suffix=".png") as f:
                        fig.savefig(f)
                        mlflow.log_artifact(local_path=f.name, artifact_path=f"{phase}/{epoch}/{artifact_name}")
                    plt.clf()

    writer.close()
    
    if phase == 'Test':
        th_prs = defaultdict(dict)
        for cond, cond_type in CONDITIONS.items():
            if cond_type != CLASS_TASK:
                continue
            for age_group in np.unique(age_groups):
                th_pr = metrics_dict['recall_0.8'][f'{cond}_{age_group}'].pretopr
                th_pr = (th_pr.th, th_pr.pr, th_pr.re)
                th_prs[cond][age_group] = th_pr
  
            th_pr = metrics_dict['recall_0.8'][f'{cond}'].pretopr
            th_pr = (th_pr.th, th_pr.pr, th_pr.re)
            th_prs[cond][None] = th_pr
            th_prs[cond][np.nan] = th_pr
                    
        with tempfile.TemporaryDirectory() as tmpdirname:
            with open(os.path.join(tmpdirname, 'th_prcs.pickle'), 'wb') as f:
                pickle.dump(th_prs, f, protocol=pickle.HIGHEST_PROTOCOL, recurse=True)

            mlflow.log_artifact(artifact_path=f"Model_onnx", local_path=f.name)


def compute_metrics(y_pred: torch.Tensor, y_label: torch.Tensor,
                    age_groups: List,
                    loss, phase: str, epoch: int, log_plots: bool = False) -> dict:
    """
    Compute metrics at the end of one epoch in the train loop
    All metrics - from torchmetrics library
    :return: dict with metrics names in keys and metrics values in values
    """
    y_label[torch.isnan(y_label)] = y_pred[torch.isnan(y_label)].to(y_label)
    metrics_dict = init_metrics(age_groups)
    metrics_values = {}
    update_metrics(y_pred, y_label, age_groups, metrics_dict)

    for metric in metrics_dict.keys():
        metrics_values[metric] = {}
        total_for_agg = 0
        for target in metrics_dict[metric].keys():
            metrics_values[metric][target] = metrics_dict[metric][target].compute()
            metrics_dict[metric][target].reset()
        avg_metric = np.nanmean([metrics_values[metric].get(cond, np.nan) for cond in HCC_GROUPS])
        if not np.isnan(avg_metric):
            metrics_values[metric]["AVG"] = avg_metric

    log_metrics(metrics_dict, age_groups, metrics_values, loss, phase, epoch, log_plots=log_plots)
    return metrics_values
