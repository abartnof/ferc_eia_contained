#!/usr/bin/env python
# coding: utf-8

# # Get y-fits from model a ANN and GBM, which will be input for the stage 2 model
# 
# __author__: Andrew Bartnof
# 
# __copyright__: Copyright 2025, Rocky Mountain Institute
# 
# __credits__: Alex Engel, Andrew Bartnof

# In[7]:


import pandas as pd
import numpy as np
import os

from keras import models, layers, regularizers, optimizers, callbacks, utils, losses, metrics
from tensorflow.keras.backend import clear_session
from tensorflow import convert_to_tensor

from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
from tqdm.notebook import tqdm


# In[8]:


def np_cleaning(X):
    X = np.clip(X, a_min=-3, a_max=3)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


# In[9]:


data_dir = '/Volumes/Extreme SSD/rematch_eia_ferc1_docker'
dir_working_model_a_training = os.path.join(data_dir, 'working_data/model_a/model_a_training')
# dir_working_model_a_training


# In[10]:


# output file
fn_y_fit_model_a = os.path.join(data_dir, 'working_data/model_z/y_fit_model_a.parquet')


# In[11]:


fn_x = os.path.join(dir_working_model_a_training, 'x.parquet')
fn_y = os.path.join(dir_working_model_a_training, 'y.parquet')
fn_id = os.path.join(dir_working_model_a_training, 'id.parquet')

X = pd.read_parquet(fn_x)
Y = pd.read_parquet(fn_y)
ID = pd.read_parquet(fn_id)


# In[14]:


params_model_a_ann = {
    'dropout_1': 0.000120,
    'dropout_2': 0.0633,
    'relu_1': 33,
    'relu_2': 20,
    'epochs': 20
}
params_model_a_gbm = {
    'num_trees':482,
    'learning_rate':0.0134,
    'min_data_in_leaf':85,
    'objective':'binary',
    'early_stopping_round':-1,
    'metrics':['binary_logloss', 'auc']
}


# In[ ]:


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
    mod_a_gbm = lgb.train(
            params = params_model_a_gbm,
            train_set=train_set
        )

    # ANN
    clear_session()
    mod_a_ann = models.Sequential()
    mod_a_ann.add(layers.Dropout(rate=params_model_a_ann["dropout_1"]))
    mod_a_ann.add(layers.Dense(units=params_model_a_ann["relu_1"], activation='relu'))    
    mod_a_ann.add(layers.Dropout(rate=params_model_a_ann["dropout_2"]))
    mod_a_ann.add(layers.Dense(units=params_model_a_ann["relu_2"], activation='relu'))   
    mod_a_ann.add(layers.Dense(1, activation='sigmoid'))
    
    mod_a_ann.compile(
        loss=losses.BinaryCrossentropy(),
        metrics=[
            metrics.BinaryCrossentropy(),
            metrics.BinaryAccuracy(), 
            metrics.AUC()
        ]
    )
        
    history = mod_a_ann.fit(
        XTrain, YTrain, epochs=params_model_a_ann['epochs'], batch_size=128,  # hard-coded here
        verbose=1
    )

    # Make predictions
    yfit_a_gbm = mod_a_gbm.predict(XTest)
    
    yfit_a_ann = mod_a_ann.predict(XTest)
    yfit_a_ann = yfit_a_ann.reshape(-1,)

    # Join ID to YFit, store together
    RelevantID = ID.loc[~is_train_mask, ['record_id_ferc1', 'record_id_eia', 'fold']].reset_index(drop=True)

    RelevantYFit = pd.DataFrame({
        'y_fit_a_ann':yfit_a_ann,
        'y_fit_a_gbm':yfit_a_gbm
    })
    
    Results = pd.concat([RelevantID, RelevantYFit], axis=1)
    results_list.append(Results)


# In[16]:


pd.concat(results_list).reset_index(drop=True).to_parquet(fn_y_fit_model_a)


# In[ ]:




