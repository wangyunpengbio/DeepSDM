import numpy as np
import yaml
import os
import torch
import logging
import sys
from pathlib import Path
from functools import partial, update_wrapper

def load_yaml(file_name):
    with open(file_name, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)
    return config

def init_seed(SEED=42):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

def init_logger(directory, log_file_name):
    formatter = logging.Formatter('\n%(asctime)s\t%(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    log_path = Path(directory, log_file_name)
    if log_path.exists():
        log_path.unlink()
    handler = logging.FileHandler(filename=log_path)
    handler.setFormatter(formatter)

    logger = logging.getLogger(log_file_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger

def wrapped_partial(func, *args, **kwargs): # 用functools包装函数，同时还能保留原来函数名
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func

class averageMeanCount():
    def __init__(self):
        self.batch_count = 0
        self.averagemean = 0.0
    def get(self):
        return self.averagemean
    def add(self, currentScore):
        self.averagemean = (self.averagemean * self.batch_count + currentScore) / (self.batch_count + 1)
        self.batch_count += 1
        return None

boundaryType2function = dict(
    BD = dict(sdfFunction='compute_sdf',lossFunction='boundary_loss'),
    BD_normalize = dict(sdfFunction='compute_sdf_normalize',lossFunction='boundary_loss'),
    HD = dict(sdfFunction='compute_dtm',lossFunction='hd_loss'),
    HD_normalize = dict(sdfFunction='compute_dtm_normalize',lossFunction='hd_loss'),
    AAAISDF_L1_product_normalize = dict(sdfFunction='compute_sdf_normalize',lossFunction='AAAI_sdf_loss'),
    AAAISDF_L1_product = dict(sdfFunction='compute_sdf',lossFunction='AAAI_sdf_loss'),
    AAAISDF_L1_normalize = dict(sdfFunction='compute_sdf_normalize',lossFunction='None'),
    AAAISDF_L1 = dict(sdfFunction='compute_sdf',lossFunction='None'),
    MultiheadSDF_L1_normalize = dict(sdfFunction='compute_sdf_normalize',lossFunction='None'),
    MultiheadSDF_L1 = dict(sdfFunction='compute_sdf',lossFunction='None'),# Fail
    MultiheadSDF_L2_normalize = dict(sdfFunction='compute_sdf_normalize',lossFunction='None'),
    MultiheadSDF_L1PlusL2_normalize = dict(sdfFunction='compute_sdf_normalize',lossFunction='None'),
    Contour_MultiheadSDF_L2_normalize = dict(sdfFunction='compute_sdf_normalize',lossFunction='compute_boundary'),
    Contour_MultiheadSDF_L2 = dict(sdfFunction='compute_sdf',lossFunction='compute_boundary'), # Fail
    Contour_MultiheadSDF_L2_normalize_innerplus = dict(sdfFunction='compute_sdf_normalize',lossFunction='compute_boundary'),
    Contour_MultiheadSDF_L2_normalize_smooth = dict(sdfFunction='compute_sdf_normalize',lossFunction='compute_boundary_smooth'),
    Contour_MultiheadSDF_L2_normalize_plain = dict(sdfFunction='compute_sdf_normalize',lossFunction='None'),
    Contour_MultiheadSDF_L2_normalize_ones = dict(sdfFunction='compute_sdf_normalize',lossFunction='None'),
    Contour_MultiheadSDF_L2_innerplus = dict(sdfFunction='compute_sdf',lossFunction='compute_boundary'),
    Contour_MultiheadSDF_L1_normalize_innerplus = dict(sdfFunction='compute_sdf_normalize',lossFunction='compute_boundary')
)