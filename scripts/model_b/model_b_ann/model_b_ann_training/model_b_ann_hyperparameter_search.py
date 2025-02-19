#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os

from keras import models, layers, regularizers, optimizers, callbacks, utils, losses, metrics
# from keras.metrics import BinaryAccuracy, AUC, BinaryCrossentropy
from tensorflow.keras.backend import clear_session
from tensorflow import convert_to_tensor

from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
from sklearn.preprocessing import StandardScaler
# utils.set_random_seed(1)


# In[2]:


data_dir = '/Volumes/Extreme SSD/rematch_eia_ferc1_docker'
dir_working_model_b_training = os.path.join(data_dir, 'working_data/model_b/model_b_training')
dir_working_model_b_training


# In[3]:


fn_x = os.path.join(dir_working_model_b_training, 'x.parquet')
fn_y = os.path.join(dir_working_model_b_training, 'y.parquet')
fn_id = os.path.join(dir_working_model_b_training, 'id.parquet')

dir_hyperparameters = dir_working_model_b_training
fn_out = os.path.join(dir_working_model_b_training, 'ann_ray_tune/model_b_ann_hp_search.csv')


# In[4]:


# fn_x = '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/working_data/model_b/train/x.parquet'
# fn_y = '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/working_data/model_b/train/y.parquet'
# fn_id = '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/working_data/model_b/train/id.parquet'

# dir_hyperparameters = '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/working_data/model_b/train'
# fn_out = '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/working_data/model_b/train/ann/grid_search.csv'


# In[5]:


def np_cleaning(X):
    X = np.clip(X, a_min=-3, a_max=3)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


# In[6]:


space = {
    'dropout_1': tune.uniform(0, 0.95),
    'dropout_2': tune.uniform(0, 0.95),
    'relu_1': tune.randint(1, 59),
    'relu_2': tune.randint(1, 30)
}

search_alg = OptunaSearch(metric=["binary_crossentropy"], mode=["min"])
search_alg = ConcurrencyLimiter(search_alg, max_concurrent=1)


# In[7]:


def fit_mod(space):
    X = pd.read_parquet(fn_x)
    Y = pd.read_parquet(fn_y)
    ID = pd.read_parquet(fn_id)
    
    # Split data into training and validation
    fold_range = np.arange(5)
    fold_variable = np.random.choice(fold_range, 1)[0]
    # fold_variable = 1
    is_train_mask = (ID['fold'] != fold_variable).values
    
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
    model.add(layers.Dense(units=space["relu_1"], activation='relu'))    
    model.add(layers.Dropout(rate=space["dropout_2"]))
    model.add(layers.Dense(units=space["relu_2"], activation='relu'))   
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
        verbose=0
    )

    train.report(
        {
            "binary_crossentropy": np.min(history.history['val_binary_crossentropy'][10:]),
            "auc": np.min(history.history['val_auc'][10:]),
            'binary_accuracy': np.min(history.history['val_binary_accuracy'][10:])
        }
    )


# In[8]:


tuner = tune.Tuner(
    fit_mod,
    tune_config=tune.TuneConfig(
        num_samples=250,
        search_alg=search_alg,
    ),
    param_space=space,
    run_config=train.RunConfig(
        storage_path=dir_hyperparameters, 
        name="ann_ray_tune"
    )
)
results = tuner.fit()


# In[10]:


Grid = results.get_dataframe().copy()
Grid.index.name = 'order'
RankedGrid = Grid.sort_values(['binary_crossentropy', 'auc'], ascending=[True, False]).reset_index()
RankedGrid.index.name = 'rank'
RankedGrid.to_csv(fn_out)
RankedGrid.head()


# In[ ]:




