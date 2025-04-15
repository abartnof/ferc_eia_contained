#!/usr/bin/env python
# coding: utf-8

# # Iterate through the tranches’ X files, and return a y_fit for each row, using the model A GBM
# 
# __author__: Andrew Bartnof
# 
# __copyright__: Copyright 2025, Rocky Mountain Institute
# 
# __credits__: Alex Engel, Andrew Bartnof

# In[11]:


import lightgbm as lgb
import numpy as np
import pandas as pd
import os 

from glob import glob

# from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm
# utils.set_random_seed(1)


# In[8]:


# Create a common location for filenames, X and y_fit
data_dir = '/Volumes/Extreme SSD/rematch_eia_ferc1_docker'
dir_x = os.path.join(data_dir, 'working_data/model_a/model_a_x')
dir_y_fit = os.path.join(data_dir, 'working_data/model_a/model_a_gbm_y_fit')

fn_list_x = glob(os.path.join(dir_x, '*.parquet'))
FN = pd.DataFrame({'dir_fn_x':fn_list_x})
FN['file_suffix'] = FN['dir_fn_x'].str.extract('(?<=x__)(.*)')

joinme_y_fit = os.path.join(dir_y_fit, 'y_fit__')
FN['dir_fn_y_fit'] = pd.Series([joinme_y_fit + f for f in FN['file_suffix'].values])


# In[9]:


# Load model
fn_model = os.path.join(data_dir, 'working_data/model_a/model_a_training/model_a_gbm.txt')
model = lgb.Booster(model_file=fn_model)


# In[12]:


for i in tqdm(FN.index):
    X = pd.read_parquet(FN['dir_fn_x'][i])
    y_fit = model.predict(X)
    YFit = pd.DataFrame(y_fit).rename(columns={0:'y_fit'})
    YFit.to_parquet(FN['dir_fn_y_fit'][i])


# In[13]:


get_ipython().system('jupyter nbconvert --to script model_a_gbm_predict.ipynb')

