#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os, re

import lightgbm as lgb

from scipy import stats


# In[22]:



def add_rank_column(xx):
    y_fit_colname = xx.columns[0]
    rank_colname = re.sub('y_fit', 'rank', y_fit_colname)
    xx[rank_colname] = xx[y_fit_colname].rank(method='dense', ascending=False)
    return(xx)


# In[61]:


data_dir = '/Volumes/Extreme SSD/rematch_eia_ferc1_docker'

fn_second_stage_model_gbm_hp = os.path.join(data_dir, 'working_data/model_second_stage/model_second_stage_training/model_second_stage_gbm_hp.csv')

fn_model_out = os.path.join(data_dir, 'working_data/model_second_stage/model_second_stage_training/model_2.txt')
fn_y_fit_2_out = os.path.join(data_dir, 'working_data/model_second_stage/model_second_stage_training/temp/y_fit_2.parquet')
fn_mod2_feature_importance = os.path.join(data_dir, 'working_data/model_second_stage/model_second_stage_training/temp/mod2_feature_importance.parquet')


# In[36]:


hp = pd.read_csv(fn_second_stage_model_gbm_hp).to_dict('list')
hp = {k:hp[k][0] for k in hp.keys()}
print(hp)


# In[27]:


fn_x_1_a = os.path.join(data_dir, 'working_data/model_second_stage/model_second_stage_training/temp/x_1_a.parquet')
fn_x_1_b = os.path.join(data_dir, 'working_data/model_second_stage/model_second_stage_training/temp/x_1_b.parquet')

fn_y_fit_1_a_ann = os.path.join(data_dir, 'working_data/model_second_stage/model_second_stage_training/temp/y_fit_1_a_ann.parquet')
fn_y_fit_1_a_gbm = os.path.join(data_dir, 'working_data/model_second_stage/model_second_stage_training/temp/y_fit_1_a_gbm.parquet')
fn_y_fit_1_b_ann = os.path.join(data_dir, 'working_data/model_second_stage/model_second_stage_training/temp/y_fit_1_b_ann.parquet')
fn_y_fit_1_b_gbm = os.path.join(data_dir, 'working_data/model_second_stage/model_second_stage_training/temp/y_fit_1_b_gbm.parquet')

fn_y = os.path.join(data_dir, 'working_data/model_a/model_a_training/y.parquet')


# In[38]:


y = pd.read_parquet(fn_y)


# In[24]:


# Load data, adding descending dense ranks to the y_fit
X1A = pd.read_parquet(fn_x_1_a).reset_index(drop=True)
X1B = pd.read_parquet(fn_x_1_b).reset_index(drop=True)
YFit1AAnn = add_rank_column(pd.read_parquet(fn_y_fit_1_a_ann).reset_index(drop=True))
YFit1AGbm = add_rank_column(pd.read_parquet(fn_y_fit_1_a_gbm).reset_index(drop=True))
YFit1BAnn = add_rank_column(pd.read_parquet(fn_y_fit_1_b_ann).reset_index(drop=True))
YFit1BGbm = add_rank_column(pd.read_parquet(fn_y_fit_1_b_gbm).reset_index(drop=True))


# In[25]:


X2 = pd.concat([
    X1A.add_prefix('x1a_'), 
    X1B.add_prefix('x1b_'), 
    YFit1AAnn, 
    YFit1AGbm, 
    YFit1BAnn, 
    YFit1BGbm
], axis=1)


# In[ ]:


train_data = lgb.Dataset(X2, y)
mod2 = lgb.train(params=hp, train_set=train_data)
mod2.save_model(fn_model_out)



y_fit = mod2.predict(X2)
pd.DataFrame(y_fit).rename(columns={0:'y_fit_2'}).to_parquet(fn_y_fit_2_out)



pd.DataFrame({'colnames':X2.columns.to_series(), 'feature_importance':mod2.feature_importance()}).reset_index(drop=True).to_parquet(fn_mod2_feature_importance)
