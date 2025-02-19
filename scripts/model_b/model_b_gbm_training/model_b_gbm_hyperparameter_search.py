#!/usr/bin/env python
# coding: utf-8

# In[1]:


import lightgbm as lgb
import numpy as np
import pandas as pd
import dask
import os 

from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
# from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from optuna.samplers import TPESampler


# In[3]:


data_dir = '/Volumes/Extreme SSD/rematch_eia_ferc1_docker'
dir_working_model_b_training = os.path.join(data_dir, 'working_data/model_b/model_b_training')
dir_working_model_b_training


# In[4]:


fn_x = os.path.join(dir_working_model_b_training, 'x.parquet')
fn_y = os.path.join(dir_working_model_b_training, 'y.parquet')
fn_id = os.path.join(dir_working_model_b_training, 'id.parquet')

dir_hyperparameters = dir_working_model_b_training
fn_out = os.path.join(dir_working_model_b_training, 'gb_ray_tune/model_b_ann_hp_search.csv')



def np_cleaning(X):
    X = np.clip(X, a_min=-3, a_max=3)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X




def fit_mod(space):
    
    # Read data
    X = pd.read_parquet(fn_x)
    Y = pd.read_parquet(fn_y)
    ID = pd.read_parquet(fn_id)
    
    # Split data into testing and training
    fold_array = np.arange(5)
    fold_variable = np.random.choice(fold_array, size=1)[0]
    
    is_train_mask = (ID['fold'] != fold_variable).values
    XTrain = X.loc[is_train_mask]
    XTest = X.loc[~is_train_mask]
    
    # Scale numeric values
    standard_scaler = StandardScaler()
    
    standard_scaler.fit(XTrain)
    XTrain = standard_scaler.transform(XTrain)
    XTest  = standard_scaler.transform(XTest)
    
    XTrain = np_cleaning(XTrain)
    XTest  = np_cleaning(XTest)
    
    # Package in training and testing objects
    train_set = lgb.Dataset(XTrain, Y.loc[is_train_mask])
    test_set  = lgb.Dataset(XTest,  Y.loc[~is_train_mask])

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


# In[7]:


space = {
    # 'num_iterations': tune.randint(1, 1000),
    'verbose':-1,
    'num_trees': tune.randint(1, 500),
    'learning_rate': tune.uniform(0.0001, 0.75),
    'min_data_in_leaf': tune.randint(1, 200),
    'objective':'binary', 
    # 'early_stopping_round':2,
    'early_stopping_round':-1,
    'metrics':['binary_logloss', 'auc']
    }


# In[8]:


# asha = ASHAScheduler(metric='binary_logloss', mode='min')

search_alg = OptunaSearch(metric="binary_logloss", mode="min")
search_alg = ConcurrencyLimiter(search_alg, max_concurrent=1)


# In[ ]:


tuner = tune.Tuner(
    fit_mod,
    tune_config=tune.TuneConfig(
        # scheduler=asha,
        num_samples=250,
        search_alg=search_alg,
    ),
    param_space=space,
    run_config=train.RunConfig(
        storage_path=dir_hyperparameters, 
        name="gb_ray_tune"
    )
)
results = tuner.fit()


# In[50]:


Grid = results.get_dataframe().copy()


# In[53]:


Grid.index.name = 'order'
RankedGrid = Grid.sort_values(['binary_logloss', 'auc'], ascending=[True, False]).reset_index()
RankedGrid.index.name = 'rank'
RankedGrid.to_csv(fn_out)


# In[60]:


RankedGrid.sort_values('binary_logloss').head(10)[['binary_logloss', 'auc', 'config/num_trees', 'config/learning_rate', 'config/min_data_in_leaf']]


# In[8]:


# experiment_path = "/Users/andrewbartnof/Documents/rmi/rematch_ferc_eia1/clean_data/model_full_gradient_boost/ray_tune/gb_ray_tune"
# restored_tuner = tune.Tuner.restore(experiment_path, trainable=fit_mod)


# In[9]:


# fn_results = '/Users/andrewbartnof/Documents/rmi/rematch_ferc_eia1/clean_data/model_full_gradient_boost/ray_tune/ray_tune_dataframe.csv'
# restored_tuner.get_results().get_dataframe().to_csv(fn_results)


# In[10]:


# !jupyter nbconvert --to script model_b_hyperparameter_search.ipynb

