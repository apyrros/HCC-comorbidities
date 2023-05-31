import random
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from inference.data_loaders.consts import ACC_NUM_COL, PAT_COL
from train.train_consts import *


def init(seed=SEED, gpu=CUDA_DEVICE):
    def seed_everything(seed: int):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    seed_everything(seed)

    os.environ['TOKENIZERS_PARALLELISM'] = 'False'
    return seed

def stratified_group_split(df_):
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    for train_idx, test_idx in splitter.split(df_.index.astype(str), df_["age_group"].astype(str), df_[PAT_COL].astype(str)):
        df_train = df_.loc[df_.index[train_idx]]
        df_test = df_.loc[df_.index[test_idx]]
        return df_train, df_test


init()
