#!/usr/bin/env python
# coding: utf-8

# # Get y-fits from model b ANN and GBM, which will be input for the stage 2 model
# 
# __author__: Andrew Bartnof
# 
# __copyright__: Copyright 2025, Rocky Mountain Institute
# 
# __credits__: Alex Engel, Andrew Bartnof

# In[1]:


import pandas as pd
import numpy as np
import os

from keras import models, layers, regularizers, optimizers, callbacks, utils, losses, metrics
from tensorflow.keras.backend import clear_session
from tensorflow import convert_to_tensor

from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
from tqdm.notebook import tqdm


# In[2]:


def np_cleaning(X):
    X = np.clip(X, a_min=-3, a_max=3)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


# In[3]:


data_dir = '/Volumes/Extreme SSD/rematch_eia_ferc1_docker'
dir_working_model_b_training = os.path.join(data_dir, 'working_data/model_b/model_b_training')
# dir_working_model_b_training


# In[4]:


# output file
fn_y_fit_model_b = os.path.join(data_dir, 'working_data/model_z/y_fit_model_b.parquet')


# In[5]:


fn_x = os.path.join(dir_working_model_b_training, 'x.parquet')
fn_y = os.path.join(dir_working_model_b_training, 'y.parquet')
fn_id = os.path.join(dir_working_model_b_training, 'id.parquet')

X = pd.read_parquet(fn_x)
Y = pd.read_parquet(fn_y)
ID = pd.read_parquet(fn_id)


# In[6]:


params_model_b_ann = {
    'dropout_1': 0.0177,
    'dropout_2': 0.00595,
    'relu_1': 56,
    'relu_2': 29,
    'epochs': 14
}
params_model_b_gbm = {
    'num_trees':266,
    'learning_rate':0.0105,
    'min_data_in_leaf':42,
    'objective':'binary',
    'early_stopping_round':-1,
    'metrics':['binary_logloss', 'auc']
}


# In[7]:


results_list = []

for fold in tqdm(ID.fold.unique()):
    
    # Create Test and Train subsets based on fold num
    is_train_mask = ID.fold != fold
    XTrain = X.loc[is_train_mask]
    XTest = X.loc[~is_train_mask]
    
    YTrain = Y.loc[is_train_mask]
    YTest = Y.loc[~is_train_mask]
    
    # Clean the X datasets, based on the training data characteristics
    standard_scaler = StandardScaler()
    standard_scaler.fit(XTrain)
    XTrain = standard_scaler.transform(XTrain)
    XTest = standard_scaler.transform(XTest)
    
    XTrain = np_cleaning(XTrain)
    XTest = np_cleaning(XTest)

    # Fit models
    # GBM
    train_set = lgb.Dataset(XTrain, YTrain)
    mod_b_gbm = lgb.train(
            params = params_model_b_gbm,
            train_set=train_set
        )

    # ANN
    clear_session()
    mod_b_ann = models.Sequential()
    mod_b_ann.add(layers.Dropout(rate=params_model_b_ann["dropout_1"]))
    mod_b_ann.add(layers.Dense(units=params_model_b_ann["relu_1"], activation='relu'))    
    mod_b_ann.add(layers.Dropout(rate=params_model_b_ann["dropout_2"]))
    mod_b_ann.add(layers.Dense(units=params_model_b_ann["relu_2"], activation='relu'))   
    mod_b_ann.add(layers.Dense(1, activation='sigmoid'))
    
    mod_b_ann.compile(
        loss=losses.BinaryCrossentropy(),
        metrics=[
            metrics.BinaryCrossentropy(),
            metrics.BinaryAccuracy(), 
            metrics.AUC()
        ]
    )
        
    history = mod_b_ann.fit(
        XTrain, YTrain, epochs=params_model_b_ann['epochs'], batch_size=128,  # hard-coded here
        verbose=1
    )

    # Make predictions
    yfit_b_gbm = mod_b_gbm.predict(XTest)
    
    yfit_b_ann = mod_b_ann.predict(XTest)
    yfit_b_ann = yfit_b_ann.reshape(-1,)

    # Join ID to YFit, store together
    RelevantID = ID.loc[~is_train_mask, ['record_id_ferc1', 'record_id_eia', 'fold']].reset_index(drop=True)

    RelevantYFit = pd.DataFrame({
        'y_fit_b_ann':yfit_b_ann,
        'y_fit_b_gbm':yfit_b_gbm
    })
    
    Results = pd.concat([RelevantID, RelevantYFit], axis=1)
    results_list.append(Results)


# In[8]:


pd.concat(results_list).reset_index(drop=True).to_parquet(fn_y_fit_model_b)

