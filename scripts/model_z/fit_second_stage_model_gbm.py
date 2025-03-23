#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os, re
import lightgbm as lgb
from scipy import stats


# In[53]:


def elt_y_fit(fn_y_fit, ID):
    # Load the y_fit, group by record_id_ferc1, return ranks
    
    YFit = pd.read_parquet(fn_y_fit)
    suffix = re.sub('y_fit_', '', YFit.columns[0])
    YFit.columns = ['y_fit']
    
    Cte = ID[['record_id_ferc1']].copy()
    Cte = pd.concat([Cte, YFit], axis=1)
    Cte['y_fit_rank'] = Cte.groupby('record_id_ferc1')['y_fit'].rank(method='dense', ascending=False)
    Cte = Cte[['y_fit', 'y_fit_rank']]
    Cte = Cte.rename(columns={'y_fit':'y_fit__' + suffix, 'y_fit_rank':'y_fit_rank__' + suffix})
    return Cte


# In[3]:


data_dir = '/Volumes/Extreme SSD/rematch_eia_ferc1_docker'

fn_second_stage_model_gbm_hp = os.path.join(data_dir, 'working_data/model_second_stage/model_second_stage_training/model_second_stage_gbm_hp.csv')

fn_model_out = os.path.join(data_dir, 'working_data/model_second_stage/model_second_stage_training/model_2.txt')
fn_y_fit_2_out = os.path.join(data_dir, 'working_data/model_second_stage/model_second_stage_training/temp/y_fit_2.parquet')
fn_mod2_feature_importance = os.path.join(data_dir, 'working_data/model_second_stage/model_second_stage_training/temp/mod2_feature_importance.parquet')


# In[4]:


hp = pd.read_csv(fn_second_stage_model_gbm_hp).to_dict('list')
hp = {k:hp[k][0] for k in hp.keys()}
print(hp)


# In[58]:


fn_x_1_a = os.path.join(data_dir, 'working_data/model_second_stage/model_second_stage_training/temp/x_1_a.parquet')
fn_x_1_b = os.path.join(data_dir, 'working_data/model_second_stage/model_second_stage_training/temp/x_1_b.parquet')

fn_y_fit_1_a_ann = os.path.join(data_dir, 'working_data/model_second_stage/model_second_stage_training/temp/y_fit_1_a_ann.parquet')
fn_y_fit_1_a_gbm = os.path.join(data_dir, 'working_data/model_second_stage/model_second_stage_training/temp/y_fit_1_a_gbm.parquet')
fn_y_fit_1_b_ann = os.path.join(data_dir, 'working_data/model_second_stage/model_second_stage_training/temp/y_fit_1_b_ann.parquet')
fn_y_fit_1_b_gbm = os.path.join(data_dir, 'working_data/model_second_stage/model_second_stage_training/temp/y_fit_1_b_gbm.parquet')

fn_id = os.path.join(data_dir, 'working_data/model_a/model_a_training/id.parquet')
fn_y = os.path.join(data_dir, 'working_data/model_a/model_a_training/y.parquet')


# In[10]:


y = pd.read_parquet(fn_y)
ID = pd.read_parquet(fn_id)

X1A = pd.read_parquet(fn_x_1_a).reset_index(drop=True)
X1B = pd.read_parquet(fn_x_1_b).reset_index(drop=True)


# In[59]:


# Load data, adding descending dense ranks to the y_fit
X1A = pd.read_parquet(fn_x_1_a).reset_index(drop=True)
X1B = pd.read_parquet(fn_x_1_b).reset_index(drop=True)

YFit1AAnn = elt_y_fit(fn_y_fit=fn_y_fit_1_a_ann, ID=ID)
YFit1AGbm = elt_y_fit(fn_y_fit=fn_y_fit_1_a_gbm, ID=ID)
YFit1BAnn = elt_y_fit(fn_y_fit=fn_y_fit_1_b_ann, ID=ID)
YFit1BGbm = elt_y_fit(fn_y_fit=fn_y_fit_1_b_gbm, ID=ID)


# In[60]:


X2 = pd.concat([
    X1A.add_prefix('x1a_').reset_index(drop=True), 
    X1B.add_prefix('x1b_').reset_index(drop=True), 
    YFit1AAnn.reset_index(drop=True), 
    YFit1AGbm.reset_index(drop=True), 
    YFit1BAnn.reset_index(drop=True), 
    YFit1BGbm.reset_index(drop=True)
], axis=1)


# In[61]:


train_data = lgb.Dataset(X2, y)
mod2 = lgb.train(params=hp, train_set=train_data)
mod2.save_model(fn_model_out)


# In[63]:


y_fit = mod2.predict(X2)
pd.DataFrame(y_fit).rename(columns={0:'y_fit_2'}).to_parquet(fn_y_fit_2_out)


# In[64]:


pd.DataFrame({'colnames':X2.columns.to_series(), 'feature_importance':mod2.feature_importance()}).reset_index(drop=True).to_parquet(fn_mod2_feature_importance)


# todo:
# - train model2
# - save model2
# - save y_fit_2
# - show component importance for model2
# - manually qc mappings

# In[65]:


get_ipython().system('jupyter nbconvert --to script fit_second_stage_model_gbm.ipynb')

