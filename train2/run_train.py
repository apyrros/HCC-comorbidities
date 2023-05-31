from typing import Tuple
from time import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import tempfile
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import onnx
from train.data.data import get_dataloaders, filter_and_process_df, test_transform
from train.model.base_model import BaseModel
from train.model.two_views_base_model import TwoViewsBaseModel
from train.model.xrv_models import *
from train.model.timm_models import *
from train.utils.metric import update_metrics, compute_metrics
from train.utils.checkpoints import *
from train.train_consts import *
from train.utils.utils import init
from inference.data_loaders.image_transform import clahe, bcet
from torch.optim.lr_scheduler import ReduceLROnPlateau
from train.data.classifier_dataset import ClassifierDataset



def train_epoch(model: nn.Module, dataloader: DataLoader, epoch: int) -> Tuple[float, dict]:
    """
    Train one epoch
    :return: tuple with loss (scalar) and metrics (dict with scalars) for train-set
    """
    train_loss = []
    tqdm_bar = tqdm(enumerate(dataloader), desc=f'Training Epoch {epoch} ', total=int(len(dataloader)))

    for i, (img, labels, age_groups) in tqdm_bar:
        loss = model.train_step(img, labels)
        train_loss.append(loss)
        mlflow.log_metric('Train_loss', float(train_loss[-1]))
        tqdm_bar.set_postfix(train_loss=np.mean(train_loss[-100:]))

        del img, labels

    return np.mean(train_loss)


def test_epoch(model: nn.Module, test_dataloader: DataLoader, epoch: int, desc: str = 'Validation', 
              log_plots: bool = False) -> \
        Tuple[float, dict, torch.Tensor, pd.DataFrame]:
    """
    Evaluate one epoch
    :return: tuple with loss (scalar) and metrics (dict with scalars) for test-set
    """
    tot_loss = []
    tqdm_bar = tqdm(enumerate(test_dataloader), desc=desc, total=int(len(test_dataloader)))
    predictions_all, labels_all, ind_all = [], [], []
    img_last = None
    for i, (img, labels, ind) in tqdm_bar:
        loss, predictions = model.predict(img, labels)
        tot_loss.append(loss)
        mlflow.log_metric(f'{desc}_loss', float(tot_loss[-1]))
        predictions_all.append(predictions)
        labels_all.append(labels)
        ind_all.extend(ind.numpy().tolist())
        tqdm_bar.set_postfix(val_loss=np.mean(tot_loss[-100:]))
        img_last = img
        del img
        

    metrics = compute_metrics(torch.cat(predictions_all), torch.cat(labels_all), test_dataloader.dataset.df.loc[ind_all]['age_group'].astype(str),
                              loss=np.mean(tot_loss), epoch=epoch,
                              phase=desc, log_plots=log_plots)

    tot_loss = np.mean(tot_loss)
    if model.scheduler is not None:
        if isinstance(model.optim, ReduceLROnPlateau):
            model.scheduler.step(tot_loss)
        else:
            model.scheduler.step(epoch)
    preds = pd.DataFrame(np.concatenate(predictions_all), index=ind_all, columns=CONDITIONS.keys())
    preds.loc[ind_all, 'ACC_NUM'] = test_dataloader.dataset.df.loc[ind_all].ACC_NUM
    return tot_loss, metrics, img_last, preds


def train(
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
        num_epochs: int = EPOCHS,
        early_stopping: int = 20,
        save_models: bool = True,
        pu_learning: bool = False,
        fold: int = 0,
):
    """
    Train loop
    """
    best_loss = np.inf
    early_stop = 0
    best_epoch = 0
    with mlflow.start_run(run_name=f'{model.model_name}_{EXPERIMENT_NAME}_{fold}', nested=True) as run:
        mlflow.log_param('batch_size', train_dataloader.batch_size)
        train_dataloader.dataset.log_params('train')
        val_dataloader.dataset.log_params('val')
        test_dataloader.dataset.log_params('test')
        mlflow.log_param('cond_weights', COND_WEIGHTS)
        mlflow.log_param('pu_learning', pu_learning)
        
        model.log_params()
        #_ = test_epoch(model, test_dataloader, desc='Test', epoch=0, log_plots=True)
        for epoch in range(num_epochs):
            _ = train_epoch(model, train_dataloader, epoch)

            val_loss, val_metrics, imgs, res = test_epoch(model, val_dataloader, epoch=epoch)

            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
                early_stop = 0
                best_epoch = epoch
                if save_models:
                    mlflow.pytorch.log_model(model, 'Model')
                   
            else:
                early_stop += 1

            if early_stop >= early_stopping:
                break
                                    
            if pu_learning and epoch > 5:
                th = 0.80 + 0.15 / max((25 - epoch), 1)
                tqdm_bar = tqdm(enumerate(train_dataloader), desc='Train prediction', total=int(len(train_dataloader)))
                img_last = None
                new_df = train_dataloader.dataset.df.copy()
                for i, (img, labels, ind) in tqdm_bar:
                    loss, predictions = model.predict(img, labels)
                    for class_id, cond in enumerate(CONDITIONS.keys()):
                        if cond in HCC_GROUPS:
                            indexes = (predictions[:, class_id] > th).numpy() 
                            new_df.loc[ind.numpy()[indexes], cond] = 1

                train_dataloader.dataset.change_df(new_df)
                for hcc in HCC_GROUPS:
                    print(train_dataloader.dataset.df[hcc].value_counts())
        
        import gc
        gc.collect()
        mlflow.log_param('best_epoch', best_epoch)
        model = mlflow.pytorch.load_model(f"runs:/{run.info.run_id}/Model")
        test_out = test_epoch(model, test_dataloader, desc='Test', epoch=0, log_plots=True)
        test_res = test_out[-1]
        onnx_model = export_to_onnx(model, imgs[:2])
        onnx_model = onnx.load_from_string(onnx_model.getvalue())
        onnx.checker.check_model(onnx_model)
        mlflow.onnx.log_model(onnx_model, f'Model_onnx')
        return test_res
            
        
def check_against_test_df(preds):
    fname = 'test_from_db.csv'
    dataset = ClassifierDataset(
            df=filter_and_process_df(fname, use_icd=True),
            conditions=CONDITIONS,
            transforms=test_transform
        )
    df = dataset.df
    
    df = df[df.ACC_NUM.isin(preds.ACC_NUM)]
    def group_accs(df_):
        return df_.groupby('ACC_NUM').agg({col:'mean' if df_[col].dtype.kind in 'biufc' else 'last' for col in df_.columns}).reset_index(drop=True)
    df = group_accs(df)
    preds = group_accs(preds)
    preds = preds.set_index('ACC_NUM').loc[df.ACC_NUM]

    labels_all = torch.Tensor(df[CONDITIONS.keys()].values)
    predictions_all = torch.Tensor(preds[[cond for cond in CONDITIONS.keys()]].values)
    metrics = compute_metrics(predictions_all, labels_all, df['age_group'].astype(str),
                          loss=0, epoch=0,
                          phase='Test', log_plots=True)

def run():
    mlflow.set_experiment("Baseline v2")

    with mlflow.start_run(run_name=f'Two views Training Full Model', nested=True) as run:
        fold = 0
        res = []
        for train_dataloader, val_dataloader, test_dataloader in get_dataloaders(
                img_size=384,  
                batch_size=64,
                use_cross_validation=True, cv_folds=5):
 
            if TWO_VIEWS:
                model = TwoViewsBaseModel(pretrain=False)
            else:
                model = BaseModel(pretrain=False)
            if (torch.cuda.device_count() > 1) and DATA_PARALLEL:
                device_ids = list(range(torch.cuda.device_count()))
                model = nn.DataParallel(model, device_ids=device_ids)
            test_df = train(model, train_dataloader, val_dataloader, test_dataloader, pu_learning=False,
                 early_stopping=10, fold=fold)
            res.append(test_df)
        res = pd.concat(res)
        check_against_test_df(res)
        with tempfile.NamedTemporaryFile(suffix=".csv") as f:
            res.to_csv(f)
            mlflow.log_artifact(local_path=f.name, artifact_path=f"out_of_fold_preds")



if __name__ == '__main__':
    init()
    run()
