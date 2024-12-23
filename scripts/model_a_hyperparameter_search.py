#!/usr/bin/env python
# coding: utf-8

# In[4]:


import lightgbm as lgb
import numpy as np
import pandas as pd
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
# from ray.tune.s|chedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter


# In[15]:


fn_train_x = '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/working_data/model_a/train/train_x.parquet'
fn_train_y = '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/working_data/model_a/train/train_y.parquet'
dir_hyperparameters = '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/working_data/model_a/train'


# In[13]:


def fit_mod(space):

    # ELT
    X = pd.read_parquet(fn_train_x)
    Y = pd.read_parquet(fn_train_y)
    # X = pd.read_parquet('/Volumes/Extreme SSD/rematch1_predictor/full_training_data/train_x.parquet')
    # Y = pd.read_parquet('/Volumes/Extreme SSD/rematch1_predictor/full_training_data/full_data_y.parquet')
    
    size_of_train_set = round(0.8 * X.shape[0])
    rows_for_train_set = np.random.choice(a=X.index, size=size_of_train_set, replace=False)
    rows_for_val_set = np.setdiff1d(X.index, rows_for_train_set)
    
    train_set = lgb.Dataset(X.loc[rows_for_train_set], Y.loc[rows_for_train_set])
    val_set = lgb.Dataset(X.loc[rows_for_val_set], Y.loc[rows_for_val_set])

    # Model
    gbm = lgb.train(
        space,
        train_set,
        valid_sets=[val_set],
    )
    binary_logloss = gbm.best_score['valid_0']['binary_logloss']
    auc = gbm.best_score['valid_0']['auc']
    train.report(
        {
            "binary_logloss": binary_logloss,
            "auc": auc
        }
    )


# In[1]:


space = {
    'num_iterations': tune.randint(1, 500),
    # 'num_rounds': tune.randint(1, 500),
    'learning_rate': tune.uniform(0.0001, 1),
    'min_data_in_leaf': tune.randint(1, 200),
    'objective':'binary', 
    # 'early_stopping_round':2,
    'metrics':['binary_logloss', 'auc']
    }


# In[17]:


# asha_scheduler = ASHAScheduler(
#     time_attr='training_iteration',
#     metric='binary_logloss',
#     mode='min',
#     max_t=1000,
#     grace_period=50,
#     reduction_factor=3,
#     brackets=1,
# )

search_alg = OptunaSearch(metric="binary_logloss", mode="min")
search_alg = ConcurrencyLimiter(search_alg, max_concurrent=2)


# In[18]:


tuner = tune.Tuner(
    fit_mod,
    tune_config=tune.TuneConfig(
        # scheduler=asha_scheduler,
        search_alg=search_alg,
        num_samples=1000
    ),
    param_space=space,
    run_config=train.RunConfig(
        storage_path=dir_hyperparameters, 
        name="gb_ray_tune"
    )
)
results = tuner.fit()


# In[ ]:


# experiment_path = "/Users/andrewbartnof/Documents/rmi/rematch_ferc_eia1/clean_data/model_full_gradient_boost/ray_tune/gb_ray_tune"
# restored_tuner = tune.Tuner.restore(experiment_path, trainable=fit_mod)


# In[ ]:


# fn_results = '/Users/andrewbartnof/Documents/rmi/rematch_ferc_eia1/clean_data/model_full_gradient_boost/ray_tune/ray_tune_dataframe.csv'
# restored_tuner.get_results().get_dataframe().to_csv(fn_results)


# In[1]:


# !jupyter nbconvert --to script model_a_hyperparameter_search.ipynb

