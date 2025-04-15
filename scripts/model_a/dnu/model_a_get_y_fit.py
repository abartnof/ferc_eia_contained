#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
import re
import glob
import keras
import lightgbm as lgb
from tqdm import tqdm



data_dir = '/Volumes/Extreme SSD/rematch_eia_ferc1_docker'
dir_working_model_a_training = os.path.join(data_dir, 'working_data/model_a/model_a_training')
dir_working_model_a_training


# In[3]:


# Load models
fn_model_a_ann = os.path.join(dir_working_model_a_training, 'model_a_ann.keras')
fn_model_a_gbm = os.path.join(dir_working_model_a_training, 'model_a_gbm.txt')

model_a_ann = keras.saving.load_model(fn_model_a_ann)
model_a_gbm = lgb.Booster(model_file=fn_model_a_gbm)


# In[88]:


# work out each filename we'll need, for the X and y_fit datafiles

dir_x = os.path.join(data_dir, 'working_data/model_a/model_a_x')
dir_id = os.path.join(data_dir, 'working_data/model_a/model_a_id')
dir_y_fit = os.path.join(data_dir, 'working_data/model_a/model_a_y_fit')

fn_mappings = os.path.join(data_dir, 'working_data/model_a/model_a_mappings.parquet')
# dir_mappings = os.path.join(data_dir, 'working_data/model_a/model_a_mappings')


# In[87]:


# Note X files, concatenate with directory
fn_x_list = glob.glob(pathname='*.parquet', root_dir=dir_x)
dir_fn_x_list = [os.path.join(dir_x, f) for f in fn_x_list]

# Extract the tranche names
tranche_list = [ re.sub('x__', '', f) for f in fn_x_list ]
tranche_list = [ re.sub('\\.parquet', '', f) for f in tranche_list ]

# Create the list of y, id filenames
fn_y_fit_list = [ re.sub('x__', 'y__', f) for f in fn_x_list ]
fn_id_list = [ re.sub('x__', 'id__', f) for f in fn_x_list ]

# Concatenate the y, id filenames with directories
dir_fn_y_fit_list = [os.path.join(dir_y_fit, f) for f in fn_y_fit_list]
dir_fn_id_list = [os.path.join(dir_id, f) for f in fn_id_list]

# Collect all full filenames + directories
FN = pd.DataFrame({
    'tranche':tranche_list,
    'x':dir_fn_x_list,
    'id':dir_fn_id_list,
    'y_fit':dir_fn_y_fit_list
    })
# FN.head()


# In[82]:


results_list = []

for i in tqdm(FN.index):
    X = pd.read_parquet(FN.loc[i, 'x'])
    ID = pd.read_parquet(FN.loc[i, 'id'])

    y_fit_gbm = model_a_gbm.predict(X)
    y_fit_ann = model_a_ann.predict(X, verbose=0).reshape(-1,)

    Framework = ID.copy()
    Framework['y_fit_model_a_gbm'] = y_fit_gbm
    Framework['y_fit_model_a_ann'] = y_fit_ann
    Framework[['y_fit_model_a_gbm', 'y_fit_model_a_ann']].to_parquet(FN.loc[i, 'y_fit'])

    FrameworkLong = Framework.melt(id_vars=['record_id_ferc1', 'record_id_eia'], var_name='variable', value_name='y_fit')
    mask = FrameworkLong.groupby(['record_id_ferc1', 'variable'])['y_fit'].idxmax()
    Results = FrameworkLong.loc[mask]
    Results.reset_index(drop=True, inplace=True)
    Results['tranche'] = FN.loc[i, 'tranche']
    Results = Results[['tranche', 'record_id_ferc1', 'record_id_eia', 'variable', 'y_fit']]
    results_list.append(Results)


# In[85]:


pd.concat(results_list, ignore_index=True).to_parquet(fn_mappings)


# In[86]:


# get_ipython().system('jupyter nbconvert --to script model_a_get_y_fit.ipynb')

