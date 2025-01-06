#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import itertools

from keras import models, layers, regularizers, optimizers, callbacks, utils, losses, metrics
from tensorflow.keras.backend import clear_session
from tensorflow import convert_to_tensor

from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm
# from tqdm.notebook import tqdm


# In[3]:


fn_x = '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/working_data/model_a/train/x.parquet'
fn_y = '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/working_data/model_a/train/y.parquet'
fn_id = '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/working_data/model_a/train/id.parquet'

# dir_hyperparameters = '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/working_data/model_a/train'
fn_grid = '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/working_data/model_a/train/ann/grid_search.csv'
fn_out = '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/working_data/model_a/train/ann/cross_validation_of_best_candidates_ann.csv'


# In[4]:


def np_cleaning(X):
    X = np.clip(X, a_min=-3, a_max=3)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


# In[5]:


X = pd.read_parquet(fn_x)
Y = pd.read_parquet(fn_y)
ID = pd.read_parquet(fn_id)


# In[6]:


rename_dict = {'config/dropout_1':'dropout_1', 'config/dropout_2':'dropout_2', 'config/relu_1':'relu_1', 'config/relu_2':'relu_2'} #, 'config/metrics':'metrics'}

Grid = pd.read_csv(fn_grid, index_col='rank')
Grid = Grid.rename(columns=rename_dict)[list(rename_dict.values())]

# Create a dictionary: punch in the rank of the model we want to use, and get the parameters back, as a dictionary
param_dict = {i:Grid.loc[i].to_dict() for i in Grid.index}


variables = [
    range(10),  # num hyperparameters to test
    range(5)  # number of folds in the ID table
]

results_list = []
for (hp_rank, fold) in tqdm(list(itertools.product(*variables))):
    
    space = param_dict[hp_rank]
    # Split data into training and validation
    is_train_mask = (ID['fold_num'] != fold).values
    
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
        XTrain, y_train, epochs=500, batch_size=128,  # hard-coded here
        validation_data=(XVal, y_val), 
        callbacks=callbacks.EarlyStopping(patience=5, start_from_epoch=10),
        verbose=1
    )
    Results = pd.DataFrame(history.history) 
    Results['hp_rank'] = hp_rank
    Results['fold'] = fold
    Results['epoch'] = Results.index + 1
    results_list.append(Results)

    # metric_dict = {
    #     "num_epochs": len( history.history['val_binary_crossentropy'] ) - 5,  # with patience set at 5, we can just find len and subtract five
    #     "binary_crossentropy": np.min(history.history['val_binary_crossentropy'][10:]),
    #     "auc": np.min(history.history['val_auc'][10:]),
    #     'binary_accuracy': np.min(history.history['val_binary_accuracy'][10:])
    #     }
    # Metrics = pd.DataFrame(metric_dict, index=range(1))
    # metrics_list.append(Metrics)


# In[23]:


CollectedMetrics = pd.concat(results_list)
CollectedMetrics.reset_index(drop=True, inplace=True)
CollectedMetrics.to_csv(fn_out, index=False)

