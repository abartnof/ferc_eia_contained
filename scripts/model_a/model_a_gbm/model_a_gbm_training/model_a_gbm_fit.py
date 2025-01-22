#!/usr/bin/env python
# coding: utf-8

# In[1]:


import lightgbm as lgb
import numpy as np
import pandas as pd
import os 

from sklearn.preprocessing import StandardScaler


# In[2]:


data_dir = '/Volumes/Extreme SSD/rematch_eia_ferc1_docker'
dir_working_model_a_training = os.path.join(data_dir, 'working_data/model_a/model_a_training')
dir_working_model_a_training


# In[5]:


fn_x = os.path.join(dir_working_model_a_training, 'x.parquet')
fn_y = os.path.join(dir_working_model_a_training, 'y.parquet')
fn_id = os.path.join(dir_working_model_a_training, 'id.parquet')
fn_model = os.path.join(dir_working_model_a_training, 'model_a_gbm.txt')
# dir_hyperparameters = '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/working_data/model_a/train'
# fn_out = '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/working_data/model_a/train/gb_ray_tune/grid_search.csv'


# In[6]:


def np_cleaning(X):
    X = np.clip(X, a_min=-3, a_max=3)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


# In[18]:


param_dict = {
    'num_trees':482,
    'learning_rate':0.0134,
    'min_data_in_leaf':85,
    'objective':'binary',
    'early_stopping_round':-1,
    'metrics':['binary_logloss', 'auc']
}


# In[10]:


X = pd.read_parquet(fn_x)
Y = pd.read_parquet(fn_y)
ID = pd.read_parquet(fn_id)


# In[14]:


# This is all done automagically by the R script that creates the new data tranches.
# We only need to do this for the final model training
standard_scaler = StandardScaler()
standard_scaler.fit(X)
XClean = standard_scaler.transform(X)
XClean = np_cleaning(XClean)


# In[19]:


train_set = lgb.Dataset(XClean, Y)
gbm = lgb.train(
        params = param_dict,
        train_set=train_set   
    )


# In[22]:


gbm.save_model(fn_model)


# In[40]:


# Optional: view the feature importances
# pd.DataFrame({'importance':gbm.feature_importance(),
#              'name':X.columns}).plot.barh(x='name', y='importance', figsize=[8, 10])


# In[26]:


X.columns.to_list()


# In[20]:


gbm.save_model(fn_model)


# In[41]:


def fit_mod(space):
    
    # Read data
    X = pd.read_parquet(fn_x)
    Y = pd.read_parquet(fn_y)
    ID = pd.read_parquet(fn_id)
    
    # Split data into testing and training
    fold_array = np.arange(5)
    fold_variable = np.random.choice(fold_array, size=1)[0]
    
    is_train_mask = (ID['fold_num'] != fold_variable).values
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


# In[42]:


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


# In[10]:


# !jupyter nbconvert --to script model_a_hyperparameter_search.ipynb

