#!/usr/bin/env python
# coding: utf-8

# # Perform raytune/optuna hyperparameter search
# 
# __author__: Andrew Bartnof
# 
# __copyright__: Copyright 2025, Rocky Mountain Institute
# 
# __credits__: Alex Engel, Andrew Bartnof

# In[10]:


import pandas as pd
import numpy as np
import os

import lightgbm as lgb

from sklearn.preprocessing import StandardScaler

from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter

from sklearn.metrics import log_loss


# In[18]:


data_dir = '/Volumes/Extreme SSD/rematch_eia_ferc1_docker'


# In[10]:


dir_hyperparameters = os.path.join(data_dir, 'working_data/model_second_stage/model_second_stage_training/gbm_raytune_2025_03_21')
fn_hyperparameters = os.path.join(dir_hyperparameters, 'gbm_grid_2025_03_21.csv')

dir_temp = os.path.join(data_dir, 'working_data/model_second_stage/model_second_stage_training/gbm_raytune_2025_03_21/temp_folder')
fn_x_train = os.path.join(dir_temp, 'x_train.parquet')
fn_x_test  = os.path.join(dir_temp, 'x_test.parquet')
fn_y_train = os.path.join(dir_temp, 'y_train.parquet')
fn_y_test  = os.path.join(dir_temp, 'y_test.parquet')


# In[11]:


space = {
    'verbose':-1,
    'num_trees': tune.randint(1, 1000),  # used to max at 500
    'learning_rate': tune.uniform(0.0001, 0.75),
    'min_data_in_leaf': tune.randint(1, 200),
    'objective':'binary', 
    # 'early_stopping_round':2,
    'early_stopping_round':-1,
    'metrics':['binary_logloss', 'auc']
    }


# In[12]:


search_alg = OptunaSearch(metric="binary_logloss", mode="min")
search_alg = ConcurrencyLimiter(search_alg, max_concurrent=1)


# In[13]:


def fit_mod(space):
    
    # Read data
    XTrain = pd.read_parquet(fn_x_train)
    XTest  = pd.read_parquet(fn_x_test)
    YTrain = pd.read_parquet(fn_y_train)
    YTest  = pd.read_parquet(fn_y_test)
    
    # Package in training and testing objects
    train_set = lgb.Dataset(XTrain, YTrain)
    test_set  = lgb.Dataset(XTest,  YTest)

    # Model
    gbm = lgb.train(
        space,
        train_set,
        valid_sets=[test_set]    
    )
    binary_logloss = gbm.best_score['valid_0']['binary_logloss']
    auc = gbm.best_score['valid_0']['auc']
    train.report(
        {
            "binary_logloss": binary_logloss,
            "auc": auc
        }
    )


# In[ ]:


tuner = tune.Tuner(
    fit_mod,
    tune_config=tune.TuneConfig(
        num_samples=500,  # 250 at prev. stages
        search_alg=search_alg,
    ),
    param_space=space,
    run_config=train.RunConfig(
        storage_path=dir_hyperparameters, 
        name="gb_ray_tune"
    )
)
results = tuner.fit()


# In[ ]:


Grid = results.get_dataframe().copy()


# In[ ]:


Grid.index.name = 'order'
RankedGrid = Grid.sort_values(['binary_logloss', 'auc'], ascending=[True, False]).reset_index()
RankedGrid.index.name = 'rank'
RankedGrid.to_csv(fn_hyperparameters)


# In[ ]:


RankedGrid.sort_values('binary_logloss').head(10)[['binary_logloss', 'auc', 'config/num_trees', 'config/learning_rate', 'config/min_data_in_leaf']]

