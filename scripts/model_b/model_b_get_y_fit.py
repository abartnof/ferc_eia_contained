#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import re
import glob
import keras
import lightgbm as lgb
from tqdm.notebook import tqdm
# from tqdm import tqdm


# In[3]:


data_dir = '/Volumes/Extreme SSD/rematch_eia_ferc1_docker'
dir_working_model_b_training = os.path.join(data_dir, 'working_data/model_b/model_b_training')
# dir_working_model_b_training


# In[4]:


# Load models
fn_model_b_ann = os.path.join(dir_working_model_b_training, 'model_b_ann.keras')
fn_model_b_gbm = os.path.join(dir_working_model_b_training, 'model_b_gbm.txt')

model_b_ann = keras.saving.load_model(fn_model_b_ann)
model_b_gbm = lgb.Booster(model_file=fn_model_b_gbm)


# In[10]:


fn_tranches = os.path.join(data_dir, 'working_data/tranches_ferc_to_eia')


# In[11]:


# work out each filename we'll need, for the X and y_fit datafiles

dir_x = os.path.join(data_dir, 'working_data/model_b/model_b_x')
# dir_id = os.path.join(data_dir, 'working_data/model_b/model_b_id')
dir_tranches = fn_tranches = os.path.join(data_dir, 'working_data/tranches_ferc_to_eia')
dir_y_fit = os.path.join(data_dir, 'working_data/model_b/model_b_y_fit')

fn_mappings = os.path.join(data_dir, 'working_data/model_b/model_b_mappings.parquet')
# dir_mappings = os.path.join(data_dir, 'working_data/model_b/model_b_mappings')


# For each type of information (X, y_fit, tranche in lieu of ID), we'll note the full export filename, and the tranche that this corresponds to. Once we've done this all 3 times, join them into an ur-table.

# In[87]:


# Note X files, concatenate with directory
fn_x_list = glob.glob(pathname='*.parquet', root_dir=dir_x)
dir_fn_x_list = [os.path.join(dir_x, f) for f in fn_x_list]

fn_id_list = [ re.sub('^.*x__', '', f) for f in fn_x_list ]
fn_id_list = [ re.sub('\\.parquet$', '', f) for f in fn_id_list ]

CteX = pd.DataFrame({'x':dir_fn_x_list, 'fn_id':fn_id_list}).set_index('fn_id')


# In[88]:


fn_tranche_list = glob.glob(pathname='*.parquet', root_dir=dir_tranches)
dir_fn_tranche_list = [os.path.join(dir_tranches, f) for f in fn_tranche_list]

fn_id_list = [ re.sub('^.*tranche__', '', f) for f in fn_tranche_list ]
fn_id_list = [ re.sub('\\.parquet$', '', f) for f in fn_id_list ]

CteTranche = pd.DataFrame({'tranche':dir_fn_tranche_list, 'fn_id':fn_id_list}).set_index('fn_id')


# In[89]:


# Create the list of y, id filenames
fn_y_fit_list = [ re.sub('x__', 'y__', f) for f in fn_x_list ]
dir_fn_y_fit_list = [os.path.join(dir_y_fit, f) for f in fn_y_fit_list]

fn_id_list = [ re.sub('^.*y__', '', f) for f in fn_y_fit_list ]
fn_id_list = [ re.sub('\\.parquet$', '', f) for f in fn_id_list ]

CteYFit = pd.DataFrame({'y_fit':dir_fn_y_fit_list, 'fn_id':fn_id_list}).set_index('fn_id')


# In[90]:


FN = CteX.join(CteTranche, how='inner').join(CteYFit, how='inner')
# FN.head()


# In[91]:


results_list = []

# for i in tqdm(FN.index):
i = FN.index[0]
X = pd.read_parquet(FN.loc[i, 'x'])
ID = pd.read_parquet(FN.loc[i, 'tranche'])


# In[94]:


y_fit_gbm = model_b_gbm.predict(X)
y_fit_ann = model_b_ann.predict(X, verbose=0).reshape(-1,)


# In[95]:


# Save two columns- y-fit for each model
Framework = ID.copy()
Framework['y_fit_model_b_gbm'] = y_fit_gbm
Framework['y_fit_model_b_ann'] = y_fit_ann
Framework[['y_fit_model_b_gbm', 'y_fit_model_b_ann']].to_parquet(FN.loc[i, 'y_fit'])


# In[ ]:


results_list = []

for i in tqdm(FN.index):
# i = FN.index[0]
    X = pd.read_parquet(FN.loc[i, 'x'])
    ID = pd.read_parquet(FN.loc[i, 'tranche'])
    
    y_fit_gbm = model_b_gbm.predict(X)
    y_fit_ann = model_b_ann.predict(X, verbose=0).reshape(-1,)
    
    # Save two columns- y-fit for each model
    Framework = ID.copy()
    Framework['y_fit_model_b_gbm'] = y_fit_gbm
    Framework['y_fit_model_b_ann'] = y_fit_ann
    Framework[['y_fit_model_b_gbm', 'y_fit_model_b_ann']].to_parquet(FN.loc[i, 'y_fit'])
    
    # Also, note the best mappings per record_id_ferc1
    FrameworkLong = Framework.melt(id_vars=['record_id_ferc1', 'record_id_eia'], var_name='variable', value_name='y_fit')
    mask = FrameworkLong.groupby(['record_id_ferc1', 'variable'])['y_fit'].idxmax()
    Results = FrameworkLong.loc[mask]
    Results.reset_index(drop=True, inplace=True)
    Results['tranche'] = FN.loc[i, 'tranche']
    Results = Results[['tranche', 'record_id_ferc1', 'record_id_eia', 'variable', 'y_fit']]
    results_list.append(Results)


# In[9]:


results_list = []

for i in tqdm(FN.index):
    X = pd.read_parquet(FN.loc[i, 'x'])
    ID = pd.read_parquet(FN.loc[i, 'id'])

    y_fit_gbm = model_b_gbm.predict(X)
    y_fit_ann = model_b_ann.predict(X, verbose=0).reshape(-1,)

    # Save two columns- y-fit for each model
    Framework = ID.copy()
    Framework['y_fit_model_b_gbm'] = y_fit_gbm
    Framework['y_fit_model_b_ann'] = y_fit_ann
    Framework[['y_fit_model_b_gbm', 'y_fit_model_b_ann']].to_parquet(FN.loc[i, 'y_fit'])

    # Also, note the best mappings per record_id_ferc1
    FrameworkLong = Framework.melt(id_vars=['record_id_ferc1', 'record_id_eia'], var_name='variable', value_name='y_fit')
    mask = FrameworkLong.groupby(['record_id_ferc1', 'variable'])['y_fit'].idxmax()
    Results = FrameworkLong.loc[mask]
    Results.reset_index(drop=True, inplace=True)
    Results['tranche'] = FN.loc[i, 'tranche']
    Results = Results[['tranche', 'record_id_ferc1', 'record_id_eia', 'variable', 'y_fit']]
    results_list.append(Results)


# In[85]:


pd.concat(results_list, ignore_index=True).to_parquet(fn_mappings)


# In[89]:


get_ipython().system('jupyter nbconvert --to script model_b_get_y_fit.ipynb')

