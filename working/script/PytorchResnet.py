# %% [markdown]
# # Pytorch Resnet
# 
# 参考Notebook：https://www.kaggle.com/a763337092/pytorch-resnet-starter-training

# %% [code] {"jupyter":{"outputs_hidden":false}}
import os
import time
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import log_loss, roc_auc_score

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

# %% [code] {"jupyter":{"outputs_hidden":false}}
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

DATA_PATH = '../input/jane-street-market-prediction/'

NFOLDS = 5
TRAIN = False

CACHE_PATH = '../input/mlp012003weights'

def save_pickle(dic, save_path):
    with open(save_path, 'wb') as f:
    # with gzip.open(save_path, 'wb') as f:
        pickle.dump(dic, f)

def load_pickle(load_path):
    with open(load_path, 'rb') as f:
    # with gzip.open(load_path, 'rb') as f:
        message_dict = pickle.load(f)
    return message_dict

# %% [code] {"jupyter":{"outputs_hidden":false}}
feat_cols = [f'feature_{i}' for i in range(130)]

target_cols = ['action', 'action_1', 'action_2', 'action_3', 'action_4']

f_mean = np.load(f'{CACHE_PATH}/f_mean_online.npy')

##### Making features
all_feat_cols = [col for col in feat_cols]
all_feat_cols.extend(['cross_41_42_43', 'cross_1_2'])

# %% [code] {"jupyter":{"outputs_hidden":false}}
feat_cols

# %% [code] {"jupyter":{"outputs_hidden":false}}


##### Model&Data fnc
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.batch_norm0 = nn.BatchNorm1d(len(all_feat_cols))
        self.dropout0 = nn.Dropout(0.2)

        dropout_rate = 0.2
        hidden_size = 256
        self.dense1 = nn.Linear(len(all_feat_cols), hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.dense2 = nn.Linear(hidden_size+len(all_feat_cols), hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.dense3 = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.dense4 = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.batch_norm4 = nn.BatchNorm1d(hidden_size)
        self.dropout4 = nn.Dropout(dropout_rate)

        self.dense5 = nn.Linear(hidden_size+hidden_size, len(target_cols))

        self.Relu = nn.ReLU(inplace=True)
        self.PReLU = nn.PReLU()
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        # self.GeLU = nn.GELU()
        self.RReLU = nn.RReLU()

    def forward(self, x):
        x = self.batch_norm0(x)
        x = self.dropout0(x)

        x1 = self.dense1(x)
        x1 = self.batch_norm1(x1)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x1 = self.LeakyReLU(x1)
        x1 = self.dropout1(x1)

        x = torch.cat([x, x1], 1)

        x2 = self.dense2(x)
        x2 = self.batch_norm2(x2)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x2 = self.LeakyReLU(x2)
        x2 = self.dropout2(x2)

        x = torch.cat([x1, x2], 1)

        x3 = self.dense3(x)
        x3 = self.batch_norm3(x3)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x3 = self.LeakyReLU(x3)
        x3 = self.dropout3(x3)

        x = torch.cat([x2, x3], 1)

        x4 = self.dense4(x)
        x4 = self.batch_norm4(x4)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x4 = self.LeakyReLU(x4)
        x4 = self.dropout4(x4)

        x = torch.cat([x3, x4], 1)

        x = self.dense5(x)

        return x

if True:
    device = torch.device("cuda:0")

    model_list = []
    tmp = np.zeros(len(feat_cols))
    for _fold in range(NFOLDS):
        torch.cuda.empty_cache()
        model = Model()
        model.to(device)
        model_weights = f"{CACHE_PATH}/online_model{_fold}.pth"
        model.load_state_dict(torch.load(model_weights))
        model.eval()
        model_list.append(model)

    import janestreet
    env = janestreet.make_env()
    env_iter = env.iter_test()

    for (test_df, pred_df) in tqdm(env_iter):
        if test_df['weight'].item() > 0:
            x_tt = test_df.loc[:, feat_cols].values
            if np.isnan(x_tt.sum()):
                x_tt = np.nan_to_num(x_tt) + np.isnan(x_tt) * f_mean

            cross_41_42_43 = x_tt[:, 41] + x_tt[:, 42] + x_tt[:, 43]
            cross_1_2 = x_tt[:, 1] / (x_tt[:, 2] + 1e-5)
            feature_inp = np.concatenate((
                x_tt,
                np.array(cross_41_42_43).reshape(x_tt.shape[0], 1),
                np.array(cross_1_2).reshape(x_tt.shape[0], 1),
            ), axis=1)

            pred = np.zeros((1, len(target_cols)))
            for model in model_list:
                pred += model(torch.tensor(feature_inp, dtype=torch.float).to(device)).sigmoid().detach().cpu().numpy() / NFOLDS
            pred = np.median(pred)
            pred_df.action = np.where(pred >= 0.5, 1, 0).astype(int)
        else:
            pred_df.action = 0
        env.predict(pred_df)