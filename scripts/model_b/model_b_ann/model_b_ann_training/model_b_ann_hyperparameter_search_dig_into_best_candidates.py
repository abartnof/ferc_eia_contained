#!/usr/bin/env python
# coding: utf-8

# # Perform cross-validation to dig into the most promising hyperparameters
# 
# __author__: Andrew Bartnof
# 
# __copyright__: Copyright 2025, Rocky Mountain Institute
# 
# __credits__: Alex Engel, Andrew Bartnof

# In[2]:


import pandas as pd
import numpy as np
import itertools
import os

from keras import models, layers, regularizers, optimizers, callbacks, utils, losses, metrics
from tensorflow.keras.backend import clear_session
from tensorflow import convert_to_tensor

from ray import train, tune
# from ray.tune.search.optuna import OptunaSearch
# from ray.tune.search import ConcurrencyLimiter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, log_loss

from tqdm import tqdm


# In[10]:


data_dir = '/Volumes/Extreme SSD/rematch_eia_ferc1_docker'
dir_working_model_b_training = os.path.join(data_dir, 'working_data/model_b/model_b_training')


# In[11]:


fn_x = os.path.join(dir_working_model_b_training, 'x.parquet')
fn_y = os.path.join(dir_working_model_b_training, 'y.parquet')
fn_id = os.path.join(dir_working_model_b_training, 'id.parquet')

dir_hyperparameters = dir_working_model_b_training
fn_hp = os.path.join(dir_working_model_b_training, 'ann_ray_tune/model_b_ann_hp_search.csv')
fn_history = os.path.join(dir_working_model_b_training, 'ann_ray_tune/history_cross_validation_of_best_candidates_ann.csv')
fn_metrics = os.path.join(dir_working_model_b_training, 'ann_ray_tune/metrics_cross_validation_of_best_candidates_ann.csv')


# In[13]:


def np_cleaning(X):
    X = np.clip(X, a_min=-3, a_max=3)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


# In[14]:


X = pd.read_parquet(fn_x)
Y = pd.read_parquet(fn_y)
ID = pd.read_parquet(fn_id)


# In[15]:


rename_dict = {'config/dropout_1':'dropout_1', 'config/dropout_2':'dropout_2', 'config/relu_1':'relu_1', 'config/relu_2':'relu_2'} #, 'config/metrics':'metrics'}

Grid = pd.read_csv(fn_hp, index_col='rank')
Grid = Grid.rename(columns=rename_dict)[list(rename_dict.values())]

# Create a dictionary: punch in the rank of the model we want to use, and get the parameters back, as a dictionary
param_dict = {i:Grid.loc[i].to_dict() for i in Grid.index}
# param_dict[0]


# In[16]:


test = False

if test:
    variables = [range(2), range(2)]
    max_epochs = 2
else:
    variables = [
        range(15),  # num hyperparameters to test
        range(5)  # number of folds in the ID table
    ]
    max_epochs = 500

history_list = []
metrics_list = []

for (hp_rank, fold) in tqdm(list(itertools.product(*variables))):
    
    space = param_dict[hp_rank]
    # Split data into training and validation
    is_train_mask = (ID['fold'] != fold).values
    
    XTrain = X.loc[is_train_mask]
    XVal = X.loc[~is_train_mask]
    y_train = Y.loc[is_train_mask, 'is_match']
    y_val = Y.loc[~is_train_mask, 'is_match']
    
    # X value processing
    standard_scaler = StandardScaler()
    standard_scaler.fit(XTrain)
    XTrain = standard_scaler.transform(XTrain)
    XVal  = standard_scaler.transform(XVal)
    
    XTrain = np_cleaning(XTrain)
    XVal  = np_cleaning(XVal)
    
    XTrain = convert_to_tensor(XTrain)
    XVal = convert_to_tensor(XVal)

    # Fit model
    clear_session()
    model = models.Sequential()
    model.add(layers.Dropout(rate=space["dropout_1"]))
    model.add(layers.Dense(units=int(space["relu_1"]), activation='relu'))    
    model.add(layers.Dropout(rate=space["dropout_2"]))
    model.add(layers.Dense(units=int(space["relu_2"]), activation='relu'))   
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        loss=losses.BinaryCrossentropy(),
        metrics=[
            metrics.BinaryCrossentropy(),
            metrics.BinaryAccuracy(), 
            metrics.AUC()
        ]
    )
        
    history = model.fit(
        XTrain, y_train, epochs=max_epochs, batch_size=128,  # hard-coded here
        validation_data=(XVal, y_val), 
        callbacks=callbacks.EarlyStopping(patience=5, start_from_epoch=10, restore_best_weights=True),
        verbose=0
    )
    
    # Store history
    History = pd.DataFrame(history.history) 
    History['hp_rank'] = hp_rank
    History['fold'] = fold
    History['epoch'] = History.index + 1
    history_list.append(History)

    # Get goodness of fit metrics on the best-scoring iteration of the model (see: callback)
    # This involves finding the best prediction per FERC record, setting those to 1, and the rest to 0
    
    y_fit = model.predict(XVal, verbose=0)
    Framework = pd.DataFrame({
        'record_id_ferc1': ID[~is_train_mask]['record_id_ferc1'],
        'y_fit': y_fit.flatten()
    })
    Framework['groupwise_max_y_fit'] = Framework.groupby('record_id_ferc1')['y_fit'].transform('max')
    Framework['y_fit_adj'] = Framework['y_fit'] == Framework['groupwise_max_y_fit']
    
    y_fit_adj = Framework['y_fit_adj'].values
    y_true = y_val.astype(bool).values
    metric_dict = {'hp_rank':hp_rank,
        'fold':fold,
        'accuracy':accuracy_score(y_true, y_fit_adj),
        'roc_auc':roc_auc_score(y_true, y_fit_adj),
        'log_loss':log_loss(y_true, y_fit_adj),
        'precision':precision_score(y_true, y_fit_adj),
        'recall':recall_score(y_true, y_fit_adj)
    }
    Metrics = pd.DataFrame(metric_dict, index=range(1))
    metrics_list.append(Metrics)


# In[19]:


CollectedHistory = pd.concat(history_list)
CollectedHistory.reset_index(drop=True, inplace=True)
CollectedHistory.to_csv(fn_history, index=False)
CollectedHistory


# In[20]:


CollectedMetrics = pd.concat(metrics_list).reset_index()
CollectedMetrics.drop('index',axis=1, inplace=True)
CollectedMetrics.to_csv(fn_metrics, index=False)
CollectedMetrics


# In[21]:


# CollectedMetrics.drop('fold', axis=1).boxplot(
#     by='hp_rank', 
#     sharey=False, 
#     grid=False, 
#     layout = (3, 2), 
#     figsize = (10, 6), 
#     meanline=True
# )

