import polars as pl
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

# (RFE)
'''

# using Recursive Feature Elimination (RFE) to reduce feature dimension

x_train = jsr_train[feature_names].fillna(0)
y_train = jsr_train[['responder_6']].to_numpy().flatten()

# Scale the features (recommended for most estimators)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
print("data has been scaled")
# Apply RFE
estimator = RandomForestRegressor(random_state=50)
print("done")
# Retain top 20 features
selector = RFE(estimator, n_features_to_select=20)

print("done")
x_train_reduced = selector.fit_transform(x_train, y_train)

print(selector.support_)
# Get the selected feature names
selected_features = x_train.columns[selector.support_]

print("Selected Features:")
print(selected_features)

# Update x_train with the selected features
x_train_reduced_df = pd.DataFrame(x_train_reduced, columns=selected_features)
'''

feature_selected = ['feature_05', 'feature_06', 'feature_07', 'feature_18', 'feature_22',
       'feature_32', 'feature_33', 'feature_36', 'feature_41', 'feature_45',
       'feature_48', 'feature_54', 'feature_56', 'feature_57', 'feature_58',
       'feature_59', 'feature_60', 'feature_68', 'feature_69', 'feature_74']

x_train = jsr_train[feature_selected].to_numpy()
mean_value = np.nanmean(x_train)
x_train = np.nan_to_num(x_train, nan=mean_value)

y_train = jsr_train[['responder_6']].to_numpy()

for c in range(1, len(os.listdir('train.parquet')) - 1):
    jsr_ = pd.read_parquet(f'train.parquet/partition_id={c}/part-0.parquet', engine='pyarrow')
    print(f"loding part-{c}")   
    jsr_ = jsr_.sample(n=50000,random_state=50)

    x_train = np.concatenate((x_train, np.nan_to_num(jsr_[feature_selected].to_numpy())), axis=0)
    y_train = np.concatenate((y_train, np.nan_to_num(jsr_[[f'responder_6']].to_numpy())), axis=0)

    del jsr_


print(f'x:{len(x_train)}')
print(f'x:{len(y_train)}')


class RF_NET(nn.Module):
    def __init__(self):
        super().__init__()
        # self.rafire_0 = nn.LSTM(input_size=80, hidden_size=424, num_layers=5, batch_first=True)
        self.rafire = nn.Sequential(

            nn.Linear(20, 1524),
            nn.BatchNorm1d(1524),
            nn.LeakyReLU(),

            nn.Linear(1524, 824),
            nn.BatchNorm1d(824),
            nn.LeakyReLU(),

            nn.Linear(824, 424),
            nn.BatchNorm1d(424),
            nn.LeakyReLU(),

            nn.Linear(424, 124),
            nn.BatchNorm1d(124),
            nn.LeakyReLU(),

            nn.Linear(124, 1)
        )

    def forward(self, x):
        # x,_=self.rafire_0(x)
        # print(x,x.shape)
        x = self.rafire(x)
        return x

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # Good for linear layers
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

rf_net = RF_NET().train().to(device)

rf_net.apply(weights_init)

if (device.type == 'cuda') and (ngpu > 1):
    rf_net = nn.DataParallel(rf_net, list(range(ngpu)))
else:
    rf_net = nn.DataParallel(rf_net).to(device)


rf_net.load_state_dict(torch.load(f"model/L1_fine_tuned_model.pth", map_location=device))

#convert data into tensor
x_train_tensor = torch.tensor(x_train).float().to(device)
y_train_tensor = torch.tensor(y_train).float().to(device)

#create dataset and Dataloader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# criterion = torch.nn.L1Loss() # Use for binary classification
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(rf_net.parameters(), lr=0.0001, betas=(0.5, 0.999))


#
#     param.requires_grad = False
#
# # Unfreeze the last layer
# for param in rf_net.rafire[-1].parameters():
#     param.requires_grad = True

num_epochs = 100  # Adjust as needed

loss_list = []

for epoch in range(num_epochs):
    rf_net.train()  # Ensure training mode is set
    running_loss = 0.0

    for inputs, labels in train_dataloader:
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = rf_net(inputs).squeeze() # Adjust output shape if needed
        loss = criterion(outputs, labels.squeeze())
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_dataloader)}")
    loss_list.append(running_loss / len(train_dataloader))

print(loss_list)

torch.save(rf_net.state_dict(), "model/L1_fine_tuned_model.pth")

def predict(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction."""
    # All the responders from the previous day are passed in at time_id == 0. We save them in a global variable for access at every time_id.
    # Use them as extra features, if you like.
    global lags_
    if lags is not None:
        lags_ = lags
    # 1. Select the required feature columns and convert to numpy array for Keras

    y_pred = []
    for data in x_test:
        y_pred += [rf_net(torch.reshape(data, (1, len(data)))).cpu().detach().numpy().reshape(-1)]

    y_pred = np.array(y_pred)

    # 3. Prepare the DataFrame for output
    predictions = test.select('row_id').with_columns(
        pl.Series("responder_6", y_pred.flatten())
    )
    print(predictions)
    # The predict function must return a DataFrame
    assert isinstance(predictions, pl.DataFrame | pd.DataFrame)
    # with columns 'row_id', 'responer_6'
    assert predictions.columns == ['row_id', 'responder_6']
    # and as many rows as the test data.
    assert len(predictions) == len(test)

    return predictions

# inference_server = jane_street_inference_server.JSInferenceServer(predict)
#
# if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
#     inference_server.serve()
# else:
#     inference_server.run_local_gateway(
#         (
#             'test.parquet',
#             'lags.parquet',
#         )
#     )