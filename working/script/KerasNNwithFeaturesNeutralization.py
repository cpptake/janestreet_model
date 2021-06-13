# %% [markdown]
# 
# # CV結果
# 
# # ver18
# ## best pram ニュートライズ有
# 
# FOLD 0
# ROC AUC:	 0.5460606911248014	 SCORE: 2289.622246284669
# FOLD 1
# ROC AUC:	 0.5594050610055324	 SCORE: 3299.472512658139
# 
# FOLD 2
# ROC AUC:	 0.5680242535452669	 SCORE: 4335.480807082125
# FOLD 3
# ROC AUC:	 0.5747448485224484	 SCORE: 3059.614618212791
# FOLD 4
# ROC AUC:	 0.5835644934134365	 SCORE: 5894.3296893748075
# 
# AUC:0.5661277105956023
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate, Lambda, GaussianNoise, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from random import choices


SEED = 1111
inference = False
cv = False

tf.random.set_seed(SEED)
np.random.seed(SEED)

#train_pickle_file = '/kaggle/input/pickling/train.csv.pandas.pickle'
train_pickle_file = '../input/janestreettraincsvpickling/train.csv.pandas.pickle'
train = pickle.load(open(train_pickle_file, 'rb'))
#train = pd.read_csv('../input/jane-street-market-prediction/train.csv')

train = train.query('date > 85').reset_index(drop = True) 
train = train[train['weight'] != 0]

train.fillna(train.mean(),inplace=True)

train['action'] = ((train['resp'].values) > 0).astype(int)
train['bias'] = 1


features = [c for c in train.columns if "feature" in c]

# %% [code] {"jupyter":{"outputs_hidden":false}}
train.head()

# %% [markdown]
# ## Feature Neutralization

# %% [code] {"jupyter":{"outputs_hidden":false}}


# code to feature neutralize

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm.auto import tqdm

def build_neutralizer(train, features, proportion, return_neut=False):
    #Builds neutralzied features, then trains a linear model to predict neutralized features from original
    #features and return the coeffs of that model.
    neutralizer = {}
    neutralized_features = np.zeros((train.shape[0], len(features)))
    target = train[['resp', 'bias']].values
    for i, f in enumerate(features):
        # obtain corrected feature
        feature = train[f].values.reshape(-1, 1)
        coeffs = np.linalg.lstsq(target, feature)[0]
        neutralized_features[:, i] = (feature - (proportion * target.dot(coeffs))).squeeze()
        
    # train model to predict corrected features
    neutralizer = np.linalg.lstsq(train[features+['bias']].values, neutralized_features)[0]
    
    if return_neut:
        return neutralized_features, neutralizer
    else:
        return neutralizer

def neutralize_array(array, neutralizer):
    neutralized_array = array.dot(neutralizer)
    return neutralized_array


def test_neutralization():
    dummy_train = train.loc[:100000, :]
    proportion = 1.0
    neutralized_features, neutralizer = build_neutralizer(dummy_train, features, proportion, True)
    dummy_neut_train = neutralize_array(dummy_train[features+['bias']].values, neutralizer)
    
#     assert np.array_equal(neutralized_features, dummy_neut_train)
    print(neutralized_features[0, :10], dummy_neut_train[0, :10])
    

test_neutralization()

# %% [markdown]
# **We can see that it almost predicts it correctly and the offset isn't that huge.**

# %% [code] {"jupyter":{"outputs_hidden":false}}

proportion = 1.0

neutralizer = build_neutralizer(train, features, proportion)
train[features] = neutralize_array(train[features+['bias']].values, neutralizer)

# %% [code] {"jupyter":{"outputs_hidden":false}}
f_mean = np.mean(train[features[1:]].values,axis=0)

resp_cols = ['resp_1', 'resp_2', 'resp_3', 'resp', 'resp_4']

X = train.loc[:, train.columns.str.contains('feature')]
#y_train = (train.loc[:, 'action'])

y = np.stack([(train[c] > 0).astype('int') for c in resp_cols]).T

# %% [markdown]
# ## Model Training

# %% [code] {"jupyter":{"outputs_hidden":false}}
def create_mlp(
    num_columns, num_labels, hidden_units, dropout_rates, label_smoothing, learning_rate
):

    inp = tf.keras.layers.Input(shape=(num_columns,))
    x = tf.keras.layers.BatchNormalization()(inp)
    x = tf.keras.layers.Dropout(dropout_rates[0])(x)
    for i in range(len(hidden_units)):
        x = tf.keras.layers.Dense(hidden_units[i])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.swish)(x)
        x = tf.keras.layers.Dropout(dropout_rates[i + 1])(x)

    x = tf.keras.layers.Dense(num_labels)(x)
    out = tf.keras.layers.Activation("sigmoid")(x)

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
        metrics=tf.keras.metrics.AUC(name="AUC"),
    )

    return model


# パラメータ

##ver17
##optuna
#epochs = 200
#batch_size = 4096
#hidden_units = [272, 304, 512]
#dropout_rates = [0.05, 0.45, 0.1, 0.3]
#label_smoothing = 1e-2
#learning_rate = 1e-3

##ver9
batch_size = 4096
hidden_units = [192, 384, 192]
dropout_rates = [0.10143786981358652, 0.2703017847244654, 0.23148340929571917, 0.2357768967777311]
label_smoothing = 1e-2
learning_rate = 1e-3
epochs = 200

# %% [markdown]
# ## Cross Validation using GroupKFold

# %% [code] {"jupyter":{"outputs_hidden":false}}
def utility_score_bincount(date, weight, resp, action): 
    count_i = len(np.unique(date))
    Pi = np.bincount(date, weight * resp * action)
    t = np.sum(Pi) / np.sqrt(np.sum(Pi ** 2)) * np.sqrt(250 / count_i)
    u = np.clip(t, 0, 6) * np.sum(Pi)
    return u

# %% [code] {"jupyter":{"outputs_hidden":false}}
train

# %% [code] {"jupyter":{"outputs_hidden":false}}
cv = True
th = 0.5

clf = create_mlp(
    len(features), 5, hidden_units, dropout_rates, label_smoothing, learning_rate
    )

models = []


if cv:

    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve, precision_recall_curve
    import gc

    # oof validation probability array
    oof_probas = np.zeros(y.shape)

    # validation indices in case of time series split
    val_idx_all = []

    # cv strategy
    N_SPLITS = 5
    gkf = GroupKFold(n_splits=N_SPLITS)
    
    
    ###kenkonishi model utility計算版
    for fold, (train_idx, val_idx) in enumerate(gkf.split(train.action.values, groups=train.date.values)):

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx].values
        y_train, y_val = y[train_idx], y[val_idx]
        
        print(f'FOLD {fold}回目',fold)
        # training and evaluation score
        c_filepath = f'./keras_nn_fold{fold}.hdf5'
        cp_callback = tf.keras.callbacks.ModelCheckpoint(c_filepath, 
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    )
        es = EarlyStopping(patience=50)
        
        clf.fit(X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=4096,
                callbacks=[cp_callback, es])
        clf.save(f'./keras_nn_fold{fold}.hdf5',fold)
        #clf.load_weights(c_filepath)
        models.append(clf)

        oof_probas[val_idx] += clf(X_val, training=False).numpy()

        score = roc_auc_score(y_val, oof_probas[val_idx])  # classification score
        
        #####################################################################################################
        valid_score = utility_score_bincount(
            date=train.iloc[val_idx].date.values,
            weight=train.iloc[val_idx].weight.values,
            resp=train.iloc[val_idx].resp.values,
            action=np.where(np.mean(oof_probas[val_idx], axis=1)>th, 1, 0))
        print(f'FOLD {fold} ROC AUC:\t {score}\t SCORE: {valid_score}')

        # deleting excess data to avoid running out of memory
        del X_train, X_val, y_train, y_val
        gc.collect()

        # appending val_idx in case of group time series split
        val_idx_all.append(val_idx)
    
    #
    #lf.save('keras_nn_cv.hdf5')
    
    # concatenation of all val_idx for further acessing
    val_idx = np.concatenate(val_idx_all)

    
else:
    
    KFOLD = 4
    modelpath = '../input/nn-with-features-neutralization-ver12/keras_nn_fold'
    modelfile = '.hdf5'
    
    for fold in range(KFOLD):
        
        modelno = fold
        bbb = modelpath + str(modelno) + modelfile
        clf.load_weights(bbb)
        models.append(clf)
        
        print('weight load Done!')

# %% [markdown]
# ## ROC AUC

# %% [code] {"jupyter":{"outputs_hidden":false}}
if cv:
    auc_oof = roc_auc_score(y[val_idx], oof_probas[val_idx])
    print(auc_oof)

# %% [markdown]
# ## Helper functions

# %% [code] {"_kg_hide-input":true,"jupyter":{"outputs_hidden":false}}
import matplotlib.pyplot as plt

def determine_action(df, thresh):
    """Determines action based on defined threshold."""
    action = (df.weight * df.resp > thresh).astype(int)
    return action

def date_weighted_resp(df):
    """Calculates the sum of weight, resp, action product."""
    cols = ['weight', 'resp', 'action']
    weighted_resp = np.prod(df[cols], axis=1)
    return weighted_resp.sum()

def calculate_t(dates_p):
    """Calculate t based on dates sum of weighted returns"""
    e_1 =  dates_p.sum() / np.sqrt((dates_p**2).sum())
    e_2 = np.sqrt(250/np.abs(len(dates_p)))
    return e_1 * e_2

def calculate_u(df, thresh):
    """Calculates utility score, and return t and u."""
    df = df.copy()

    # calculates sum of dates weighted returns
    dates_p = df.groupby('date').apply(date_weighted_resp)
        
    # calculate t
    t = calculate_t(dates_p)
    return t, min(max(t, 0), 6) * dates_p.sum()



def plot_roc_curve(fpr, tpr, label=None):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')  # dashed diagonal
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid()
    
    
def plot_precision_recall_curve(precisions, recalls, thresholds):
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.xlabel('Thresholds')
    plt.legend(loc='lower left')
    plt.grid()


    
def plot_thresh_u_t(df, oof):
    threshs = np.linspace(0, 1, 1000)
    ts = []
    us = []
    
    for thresh in threshs:
        df['action'] = np.where(oof >= thresh, 1, 0)
        t, u = calculate_u(df, thresh)
        ts.append(t)
        us.append(u)
        
    # change nans into 0
    ts = np.array(ts)
    us = np.array(us)
    ts = np.where(np.isnan(ts), 0.0, ts)
    us = np.where(np.isnan(us), 0.0, us)
    
    tmax = np.argmax(ts)
    umax = np.argmax(us)
    
    print(f'Max Utility Score: {us[umax]}')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(threshs, ts)
    axes[0].set_title('Different t scores by threshold')
    axes[0].set_xlabel('Threshold')
    axes[0].axvline(threshs[tmax])

    axes[1].plot(threshs, us)
    axes[1].set_title('Different u scores by threshold')
    axes[1].set_xlabel('Threshold')
    axes[1].axvline(threshs[umax], color='r', linestyle='--', linewidth=1.2)
    
    print(f'Optimal Threshold: {threshs[umax]}')
    
    return threshs[umax]

# %% [markdown]
# ## ROC Curve

# %% [code] {"jupyter":{"outputs_hidden":false}}
if cv:
    fpr, tpr, thresholds = roc_curve(y[val_idx, 4], oof_probas[val_idx, 4])    
    plot_roc_curve(fpr, tpr, 'NN')

# %% [markdown]
# ## Precision/Recall Curve

# %% [code] {"jupyter":{"outputs_hidden":false}}
if cv:
    precisions, recalls, thresholds = precision_recall_curve(y[val_idx, 4], oof_probas[val_idx, 4])
    plot_precision_recall_curve(precisions, recalls, thresholds)

# %% [code] {"jupyter":{"outputs_hidden":false}}
#長いので0.506で統一

opt_thresh = 0.506
print(opt_thresh)

f = np.median
models = models[-3:]

import janestreet
env = janestreet.make_env()
for (test_df, pred_df) in tqdm(env.iter_test()):
    if test_df['weight'].item() > 0:
        x_tt = test_df.loc[:, features].values
        if np.isnan(x_tt[:, 1:].sum()):
            x_tt[:, 1:] = np.nan_to_num(x_tt[:, 1:]) + np.isnan(x_tt[:, 1:]) * f_mean
        
        ##ニュートライズする場合
        x_tt = np.append(x_tt, [[1]], axis=1)  # add bias term
        x_tt = neutralize_array(x_tt, neutralizer)
        
        pred = np.mean([model(x_tt, training = False).numpy() for model in models],axis=0)
        pred = f(pred)
        pred_df.action = np.where(pred >= opt_thresh, 1, 0).astype(int)
    else:
        pred_df.action = 0
    env.predict(pred_df)