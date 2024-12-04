import polars as pl
from IPython.core.pylabtools import figsize
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from kaggle_evaluation import jane_street_inference_server,jane_street_gateway

import os
import random
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.transforms.functional as RF
from PIL import Image
import numpy as np
import pandas as pd
import random,cv2,os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation

seed = 999
# print("Random Seed: ", seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

workers = 2
nz = 100
beta1 = 0.5
lr = 0.0001
ngpu = torch.cuda.device_count()
ngf,nc = 3,3
ndf = 64

device = torch.device("cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

jsr_ = pd.read_parquet(f'test.parquet/date_id=0/part-0.parquet', engine='pyarrow')

jsr_df = pd.DataFrame(jsr_)

x_test = torch.tensor(np.nan_to_num(jsr_[[f'feature_{str(i).zfill(2)}' for i in range(0, 79)]].to_numpy())).float().to(device)
# PCA reduce
print(len(os.listdir('test.parquet')) - 1)

jsr_train = pd.read_parquet(f'train.parquet/partition_id=0/part-0.parquet', engine='pyarrow')
x_train = pd.DataFrame(jsr_train[[f'feature_{str(i).zfill(2)}' for i in range(0, 79)]])

x_train = x_train.loc[:,x_train.isnull().mean() < 0.99]
feature_names = x_train.columns.to_numpy()

jsr_train = jsr_train.sample(n=50000,random_state=50)

feature_selected = ['feature_18', 'feature_19', 'feature_20',  'feature_22',
       'feature_32', 'feature_23', 'feature_24', 'feature_25', 'feature_28',
       'feature_48', 'feature_32', 'feature_33', 'feature_34', 'feature_35',
       'feature_59', 'feature_36', 'feature_37', 'feature_38', 'feature_39',
       'feature_40', 'feature_41', 'feature_42', 'feature_43', 'feature_44',
       'feature_45', 'feature_46', 'feature_50', 'feature_51', 'feature_52',
       'feature_53', 'feature_54', 'feature_55', 'feature_56', 'feature_57',
       'feature_61', 'feature_65', 'feature_66']


x_train = pd.DataFrame(jsr_train[feature_selected])
# x_train.hist(bins=20,figsize=(15,15))
# plt.tight_layout()
# Correlation heatmap
corr = x_train.corr()  # Get correlation matrix
plt.figure(figsize=(20,20))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Features Correlations",fontsize=50)
plt.show()

feature_selected = ['feature_05', 'feature_06', 'feature_07', 'feature_18', 'feature_22',
       'feature_32', 'feature_33', 'feature_36', 'feature_41', 'feature_45',
       'feature_48', 'feature_54', 'feature_56', 'feature_57', 'feature_58',
       'feature_59', 'feature_60', 'feature_68', 'feature_69', 'feature_74']

x_train = jsr_train[feature_selected].to_numpy()
mean_value = np.nanmean(x_train)
x_train = np.nan_to_num(x_train, nan=mean_value)

y_train = jsr_train[['responder_6']].to_numpy()
