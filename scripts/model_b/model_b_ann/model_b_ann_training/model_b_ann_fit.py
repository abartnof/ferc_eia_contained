#!/usr/bin/env python
# coding: utf-8

# # Fit model
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
# from keras.metrics import BinaryAccuracy, AUC, BinaryCrossentropy
from tensorflow.keras.backend import clear_session
from tensorflow import convert_to_tensor

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

fn_model = os.path.join(dir_working_model_b_training, 'model_b_ann.keras')


# In[4]:


fn_x_1_b_out = os.path.join(data_dir, 'working_data/model_second_stage/model_second_stage_training/temp/x_1_b.parquet')
fn_y_fit_1_b_ann = os.path.join(data_dir, 'working_data/model_second_stage/model_second_stage_training/temp/y_fit_1_b_ann.parquet')


# In[5]:


def np_cleaning(X):
    X = np.clip(X, a_min=-3, a_max=3)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


# In[6]:


fn_params = os.path.join(dir_working_model_b_training, 'model_b_ann_hp.csv')
params = pd.read_csv(fn_params).to_dict(orient='list')
params = {k:params[k][0] for k in params.keys()}

# params['metrics'] = ['binary_logloss', 'auc']
print(params)


# In[7]:


# params = {
#     'dropout_1': 0.0177,
#     'dropout_2': 0.00595,
#     'relu_1': 56,
#     'relu_2': 29,
#     'epochs': 14
# }


# In[8]:


X = pd.read_parquet(fn_x)
Y = pd.read_parquet(fn_y)
ID = pd.read_parquet(fn_id)


# In[9]:


# This is all done automagically by the R script that creates the new data tranches.
# We only need to do this for the final model training
standard_scaler = StandardScaler()
standard_scaler.fit(X)
XClean = standard_scaler.transform(X)
XClean = np_cleaning(XClean)

pd.DataFrame(XClean).to_parquet(fn_x_1_b_out)

XClean = convert_to_tensor(XClean)


# In[10]:


clear_session()
model = models.Sequential()
model.add(layers.Dropout(rate=params["dropout_1"]))
model.add(layers.Dense(units=int(params["relu_1"]), activation='relu'))    
model.add(layers.Dropout(rate=params["dropout_2"]))
model.add(layers.Dense(units=int(params["relu_2"]), activation='relu'))   
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
    XClean, Y, epochs=int(params['epochs']), batch_size=128,  # hard-coded here
    verbose=1
)


# In[11]:


model.save(fn_model)


# In[13]:


y_fit = model.predict(XClean)


# In[14]:


pd.DataFrame(y_fit).rename(columns={0:'y_fit_1_b_ann'}).to_parquet(fn_y_fit_1_b_ann)


# In[ ]:


get_ipython().system('jupyter nbconvert --to script model_b_ann_fit.ipynb')

