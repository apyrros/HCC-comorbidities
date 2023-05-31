import os
import torch
import mlflow

EXPERIMENT_NAME = "test"
CUDA_DEVICE = 1
SEED = 43

#from inference.post_processing.raw_pred_to_precision import ThresholdToPrecision
remote_server_uri = "http://localhost:5000"
mlflow.set_tracking_uri(remote_server_uri)

TRAIN_SERVICE_PATH = os.environ.get('TRAIN_SERVICE_PATH')

CACHE_DATASET = True

if TRAIN_SERVICE_PATH is None:
    TRAIN_SERVICE_PATH = os.path.abspath(os.getcwd())
    
DATA_FILES_PATH = os.environ.get('DATA_FILES_PATH', os.path.join(TRAIN_SERVICE_PATH, 'data_files'))

CHECKPOINT_DIR = "checkpoints"
NUM_WORKERS = 8
DEVICE = torch.device(f'cuda:{CUDA_DEVICE}' if torch.cuda.is_available() else 'cpu')
DATA_PARALLEL = False

TWO_VIEWS = True
SIZE = 256
AGE_NORM = 100
RAF_NORM = 10

IMAGE_PATH_COL = 'img_path'
ORIENT_COL = 'img_orient'

CLASS_TASK = 'class'
REG_TASK = 'reg'
HCC_GROUPS = ['HCC18', 'HCC22', 'HCC85', 'HCC96', 'HCC108', 'HCC111']
CONDITIONS = {
    # 'GENDER': CLASS_TASK, 
    **{hcc: CLASS_TASK for hcc in HCC_GROUPS},
    'AGE': REG_TASK, 
    'RAF': REG_TASK,
    'bmi': REG_TASK,
    #'sdi': REG_TASK,
    'a1c': REG_TASK,
    #  'RACE_WHITE': CLASS_TASK,
    #  'RACE_BLACK': CLASS_TASK,
    #  'RACE_ASIAN': CLASS_TASK,
    # 'LANG_ENG': CLASS_TASK
    
             }
COND_WEIGHTS = {
    # 'bmi': 0.3,
    # 'a1c': 0.3,
    # 'sdi': 0.3,
    # 'RACE_WHITE': 0.25,
    # 'RACE_BLACK': 0.25,
    # 'RACE_ASIAN': 0.25,
    # 'LANG_ENG': 0.25,
}

CONDITIONS_FOR_METRIC_AGG = HCC_GROUPS
THRESHOLD = 0.5

PRETRAIN = "/storage1/data/model_best.pth"

EPOCHS = 100
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64


SPLIT_DATE = "2021.01.01"
