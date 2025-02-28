#!/usr/bin/env python
# coding: utf-8

# In[3]:


import lightgbm as lgb
import numpy as np
import pandas as pd
import os 

from sklearn.preprocessing import StandardScaler


# In[4]:


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


# In[16]:


fn_params = os.path.join(dir_working_model_a_training, 'model_a_gbm_hp.csv')
params = pd.read_csv(fn_params).to_dict(orient='list')
params = {k:params[k][0] for k in params.keys()}

params['metrics'] = ['binary_logloss', 'auc']
print(params)


# In[17]:


# param_dict = {
#     'num_trees':482,
#     'learning_rate':0.0134,
#     'min_data_in_leaf':85,
#     'objective':'binary',
#     'early_stopping_round':-1,
#     'metrics':['binary_logloss', 'auc']
# }


# In[ ]:


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


# In[11]:


train_set = lgb.Dataset(XClean, Y)
gbm = lgb.train(
        params = params,
        train_set=train_set   
    )


# In[22]:


gbm.save_model(fn_model)


# In[41]:


# Optional: view the feature importances
# pd.DataFrame({'importance':gbm.feature_importance(),
#              'name':X.columns}).plot.barh(x='name', y='importance', figsize=[8, 10])


# In[42]:


# !jupyter nbconvert --to script model_a_gbm_fit.ipynb

