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

from sklearn.preprocessing import StandardScaler
# utils.set_random_seed(1)


# In[2]:


data_dir = '/Volumes/Extreme SSD/rematch_eia_ferc1_docker'
dir_working_model_a_training = os.path.join(data_dir, 'working_data/model_a/model_a_training')
dir_working_model_a_training


# In[20]:


fn_x = os.path.join(dir_working_model_a_training, 'x.parquet')
fn_y = os.path.join(dir_working_model_a_training, 'y.parquet')
fn_id = os.path.join(dir_working_model_a_training, 'id.parquet')

fn_model = os.path.join(dir_working_model_a_training, 'model_a_ann.keras')


# In[4]:


def np_cleaning(X):
    X = np.clip(X, a_min=-3, a_max=3)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


# In[10]:


params = {
    'dropout_1': 0.000120,
    'dropout_2': 0.0633,
    'relu_1': 33,
    'relu_2': 20,
    'epochs': 20
}


# In[6]:


X = pd.read_parquet(fn_x)
Y = pd.read_parquet(fn_y)
ID = pd.read_parquet(fn_id)


# In[7]:


# This is all done automagically by the R script that creates the new data tranches.
# We only need to do this for the final model training
standard_scaler = StandardScaler()
standard_scaler.fit(X)
XClean = standard_scaler.transform(X)
XClean = np_cleaning(XClean)
XClean = convert_to_tensor(XClean)


# In[11]:


clear_session()
model = models.Sequential()
model.add(layers.Dropout(rate=params["dropout_1"]))
model.add(layers.Dense(units=params["relu_1"], activation='relu'))    
model.add(layers.Dropout(rate=params["dropout_2"]))
model.add(layers.Dense(units=params["relu_2"], activation='relu'))   
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
    XClean, Y, epochs=params['epochs'], batch_size=128,  # hard-coded here
    verbose=1
)


# model
