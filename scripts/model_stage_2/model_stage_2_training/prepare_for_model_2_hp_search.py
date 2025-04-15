#!/usr/bin/env python
# coding: utf-8

# # Split and prepare the input data for the models
# 
# __author__: Andrew Bartnof
# 
# __copyright__: Copyright 2025, Rocky Mountain Institute
# 
# __credits__: Alex Engel, Andrew Bartnof

# Steps here:
# - Subset training data to 2/5 of the folds
# - fit all four first-order models on this 2/5
# - Subsequently, we'll need to train a GBM model on 2/5 of the (remaining) folds, and to test on the last 1/5
#     - These training and testing sets for the second-tier model will need the __feature-engineered__ input X files for each model (A and B), as well as the __y-fit__ for each of the input models (A, B x ANN, GBM). Also, add __descending-rank__ for each prediction, where 1.0 is 1
#     - It's easier if we pre-process this bit, and save these training and testing sets in a 'temp folder', and then in a second script, search for hyperparameters

# In[1]:


import pandas as pd
import numpy as np
import os

from keras import models, layers, regularizers, optimizers, callbacks, utils, losses, metrics
from tensorflow.keras.backend import clear_session
from tensorflow import convert_to_tensor

import lightgbm as lgb

from sklearn.preprocessing import StandardScaler

# from ray import train, tune
# from ray.tune.search.optuna import OptunaSearch
# from ray.tune.schedulers import ASHAScheduler
# from ray.tune.search import ConcurrencyLimiter

from scipy import stats


# # Establish file locations, define functions

# In[2]:


data_dir = '/Volumes/Extreme SSD/rematch_eia_ferc1_docker'
dir_working_model_a_training = os.path.join(data_dir, 'working_data/model_a/model_a_training')
dir_working_model_b_training = os.path.join(data_dir, 'working_data/model_b/model_b_training')


# In[3]:


fn_x_a = os.path.join(dir_working_model_a_training, 'x.parquet')
fn_y_a = os.path.join(dir_working_model_a_training, 'y.parquet')
fn_id_a = os.path.join(dir_working_model_a_training, 'id.parquet')

fn_x_b = os.path.join(dir_working_model_b_training, 'x.parquet')
fn_y_b = os.path.join(dir_working_model_b_training, 'y.parquet')
fn_id_b = os.path.join(dir_working_model_b_training, 'id.parquet')


# In[4]:


# (pd.read_parquet(fn_id_a) == pd.read_parquet(fn_id_b)).mean()


# In[5]:


dir_temp = os.path.join(data_dir, 'working_data/model_second_stage/model_second_stage_training/gbm_raytune_2025_03_21/temp_folder')

fn_model_a_ann = os.path.join(dir_temp, 'model_a_ann.keras')
fn_model_b_ann = os.path.join(dir_temp, 'model_b_ann.keras')
fn_model_a_gbm = os.path.join(dir_temp, 'model_a_gbm.txt')
fn_model_b_gbm = os.path.join(dir_temp, 'model_b_gbm.txt')

fn_x_train = os.path.join(dir_temp, 'x_train.parquet')
fn_x_test  = os.path.join(dir_temp, 'x_test.parquet')
fn_y_train = os.path.join(dir_temp, 'y_train.parquet')
fn_y_test  = os.path.join(dir_temp, 'y_test.parquet')

fn_split_characteristics = os.path.join(dir_temp, 'split_characteristics.csv')


# In[6]:


def np_cleaning(X):
    X = np.clip(X, a_min=-3, a_max=3)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X

def prep_x(fit_scaler, X):
    X = fit_scaler.transform(X)
    X = np_cleaning(X)
    # X = convert_to_tensor(X)
    return(X)
    
def add_grouped_rank_to_y_fit(ID, mask, y_fit):
    Cte = ID.loc[mask, ['record_id_ferc1']].copy()
    Cte['y_fit'] = y_fit
    Cte['rank'] = Cte.groupby('record_id_ferc1')['y_fit'].rank(method='dense', ascending=False)
    Cte = Cte[['y_fit', 'rank']]
    return Cte


# In[7]:


fn_model_a_ann_hp = os.path.join(data_dir, 'working_data/model_a/model_a_training/model_a_ann_hp.csv')
params_a_ann = pd.read_csv(fn_model_a_ann_hp).to_dict(orient='list')
params_a_ann = {k:params_a_ann[k][0] for k in params_a_ann.keys()}


# In[8]:


fn_model_a_gbm_hp = os.path.join(data_dir, 'working_data/model_a/model_a_training/model_a_gbm_hp.csv')
params_a_gbm = pd.read_csv(fn_model_a_gbm_hp).to_dict(orient='list')
params_a_gbm = {k:params_a_gbm[k][0] for k in params_a_gbm.keys()}
params_a_gbm['metrics'] = ['binary_logloss', 'auc']


# In[9]:


fn_model_b_ann_hp = os.path.join(data_dir, 'working_data/model_b/model_b_training/model_b_ann_hp.csv')
params_b_ann = pd.read_csv(fn_model_b_ann_hp).to_dict(orient='list')
params_b_ann = {k:params_b_ann[k][0] for k in params_b_ann.keys()}


# In[10]:


fn_model_b_gbm_hp = os.path.join(data_dir, 'working_data/model_b/model_b_training/model_b_gbm_hp.csv')
params_b_gbm = pd.read_csv(fn_model_b_gbm_hp).to_dict(orient='list')
params_b_gbm = {k:params_b_gbm[k][0] for k in params_b_gbm.keys()}
params_b_gbm['metrics'] = ['binary_logloss', 'auc']


# # Model A

# In[11]:


# Model A- data prep

X_a = pd.read_parquet(fn_x_a)
Y_a = pd.read_parquet(fn_y_a)
ID_a = pd.read_parquet(fn_id_a)


# In[12]:


# Create 3 folds: fit premier models, train secondary model, test secondary model
fold_values = ID_a.fold.unique().tolist()
np.random.shuffle(fold_values)
values_for_premier_model_fits = fold_values[0:2]
values_for_secondary_model_train = fold_values[2:4]
values_for_secondary_model_test = fold_values[4]

is_premier_model_fits    = np.isin(element=ID_a.fold.values, test_elements=values_for_premier_model_fits)
is_secondary_model_test  = np.isin(element=ID_a.fold.values, test_elements=values_for_secondary_model_test)
is_secondary_model_train = np.isin(element=ID_a.fold.values, test_elements=values_for_secondary_model_train)

frames = [
    pd.DataFrame({'name':'premier_model_fits', 'fold':values_for_premier_model_fits}),
    pd.DataFrame({'name':'secondary_model_train', 'fold':values_for_secondary_model_train}),
    pd.DataFrame({'name':'secondary_model_test', 'fold':[values_for_secondary_model_test]})
]
SplitCharacteristics = pd.concat(frames, ignore_index=True)
SplitCharacteristics


# In[13]:


YPremierModelFits = Y_a.loc[is_premier_model_fits]

YSecondaryModelTrain = Y_a.loc[is_secondary_model_train]
YSecondaryModelTest = Y_a.loc[is_secondary_model_test]


# In[14]:


# Model A Premier Model, clean X

XAPremierModelFits    = X_a.loc[is_premier_model_fits]
XASecondaryModelTrain = X_a.loc[is_secondary_model_train]
XASecondaryModelTest  = X_a.loc[is_secondary_model_test]

standard_scaler_a = StandardScaler()
standard_scaler_a = standard_scaler_a.fit(XAPremierModelFits)

XAPremierModelFits    = prep_x(fit_scaler=standard_scaler_a, X=XAPremierModelFits)
XASecondaryModelTrain = prep_x(fit_scaler=standard_scaler_a, X=XASecondaryModelTrain)
XASecondaryModelTest  = prep_x(fit_scaler=standard_scaler_a, X=XASecondaryModelTest)


# ## ANN A

# In[15]:


clear_session()
model_a_ann = models.Sequential()
model_a_ann.add(layers.Dropout(rate=params_a_ann["dropout_1"]))
model_a_ann.add(layers.Dense(units=int(params_a_ann["relu_1"]), activation='relu'))    
model_a_ann.add(layers.Dropout(rate=params_a_ann["dropout_2"]))
model_a_ann.add(layers.Dense(units=int(params_a_ann["relu_2"]), activation='relu'))   
model_a_ann.add(layers.Dense(1, activation='sigmoid'))

model_a_ann.compile(
    loss=losses.BinaryCrossentropy(),
    metrics=[
        metrics.BinaryCrossentropy(),
        metrics.BinaryAccuracy(), 
        metrics.AUC()
    ]
)
    
history_a_ann = model_a_ann.fit(
    convert_to_tensor(XAPremierModelFits), YPremierModelFits, epochs=int(params_a_ann['epochs']), batch_size=128,
    verbose=1
)


# ## GBM A

# In[ ]:


train_set = lgb.Dataset(XAPremierModelFits, YPremierModelFits)
model_a_gbm = lgb.train(
        params = params_a_gbm,
        train_set=train_set   
    )


# # Model B

# In[ ]:


# Model B- data prep

X_b = pd.read_parquet(fn_x_b)
Y_b = pd.read_parquet(fn_y_b)
ID_b = pd.read_parquet(fn_id_b)

XBPremierModelFits = X_b.loc[is_premier_model_fits]
XBSecondaryModelTest = X_b.loc[is_secondary_model_test]
XBSecondaryModelTrain = X_b.loc[is_secondary_model_train]


# In[ ]:


# Model B Premier Model, clean X

XBPremierModelFits    = X_b.loc[is_premier_model_fits]
XBSecondaryModelTrain = X_b.loc[is_secondary_model_train]
XBSecondaryModelTest  = X_b.loc[is_secondary_model_test]

standard_scaler_b = StandardScaler()
standard_scaler_b = standard_scaler_b.fit(XBPremierModelFits)

XBPremierModelFits    = prep_x(fit_scaler=standard_scaler_b, X=XBPremierModelFits)
XBSecondaryModelTrain = prep_x(fit_scaler=standard_scaler_b, X=XBSecondaryModelTrain)
XBSecondaryModelTest  = prep_x(fit_scaler=standard_scaler_b, X=XBSecondaryModelTest)


# ## ANN B

# In[ ]:


clear_session()
model_b_ann = models.Sequential()
model_b_ann.add(layers.Dropout(rate=params_b_ann["dropout_1"]))
model_b_ann.add(layers.Dense(units=int(params_b_ann["relu_1"]), activation='relu'))    
model_b_ann.add(layers.Dropout(rate=params_b_ann["dropout_2"]))
model_b_ann.add(layers.Dense(units=int(params_b_ann["relu_2"]), activation='relu'))   
model_b_ann.add(layers.Dense(1, activation='sigmoid'))

model_b_ann.compile(
    loss=losses.BinaryCrossentropy(),
    metrics=[
        metrics.BinaryCrossentropy(),
        metrics.BinaryAccuracy(), 
        metrics.AUC()
    ]
)
    
history_b_ann = model_b_ann.fit(
    convert_to_tensor(XBPremierModelFits), YPremierModelFits, epochs=int(params_b_ann['epochs']), batch_size=128,
    verbose=1
)


# ## GBM B

# In[ ]:


train_set = lgb.Dataset(XBPremierModelFits, YPremierModelFits)
model_b_gbm = lgb.train(
        params = params_b_gbm,
        train_set=train_set   
    )


# # Save data to temporary folder

# In[ ]:


# Write models, split characteristics to disk

SplitCharacteristics.to_csv(fn_split_characteristics)
model_a_ann.save(filepath=fn_model_a_ann)
model_b_ann.save(filepath=fn_model_b_ann)
model_a_gbm.save_model(fn_model_a_gbm)
model_b_gbm.save_model(fn_model_b_gbm)


# In[ ]:


y_fit_train_a_ann = model_a_ann.predict(convert_to_tensor(XASecondaryModelTrain))
y_fit_train_b_ann = model_b_ann.predict(convert_to_tensor(XBSecondaryModelTrain))

y_fit_test_a_ann = model_a_ann.predict(convert_to_tensor(XASecondaryModelTest))
y_fit_test_b_ann = model_b_ann.predict(convert_to_tensor(XBSecondaryModelTest))


# In[ ]:


y_fit_train_a_gbm = model_a_gbm.predict(XASecondaryModelTrain)
y_fit_train_b_gbm = model_b_gbm.predict(XBSecondaryModelTrain)

y_fit_test_a_gbm = model_a_gbm.predict(XASecondaryModelTest)
y_fit_test_b_gbm = model_b_gbm.predict(XBSecondaryModelTest)


# In[ ]:


XSecondaryModelTrain = np.hstack([
    XASecondaryModelTrain, 
    XBSecondaryModelTrain,
    
    add_grouped_rank_to_y_fit(ID=ID_a, mask=is_secondary_model_train, y_fit=y_fit_train_a_ann),
    add_grouped_rank_to_y_fit(ID=ID_a, mask=is_secondary_model_train, y_fit=y_fit_train_a_gbm),
    
    add_grouped_rank_to_y_fit(ID=ID_b, mask=is_secondary_model_train, y_fit=y_fit_train_b_ann),
    add_grouped_rank_to_y_fit(ID=ID_b, mask=is_secondary_model_train, y_fit=y_fit_train_b_gbm),

])


# In[ ]:


XSecondaryModelTest = np.hstack([
    XASecondaryModelTest, 
    XBSecondaryModelTest,
    
    add_grouped_rank_to_y_fit(ID=ID_a, mask=is_secondary_model_test, y_fit=y_fit_test_a_ann),
    add_grouped_rank_to_y_fit(ID=ID_a, mask=is_secondary_model_test, y_fit=y_fit_test_a_gbm),
    
    add_grouped_rank_to_y_fit(ID=ID_b, mask=is_secondary_model_test, y_fit=y_fit_test_b_ann),
    add_grouped_rank_to_y_fit(ID=ID_b, mask=is_secondary_model_test, y_fit=y_fit_test_b_gbm),

])


# In[ ]:


# XSecondaryModelTrain = np.hstack([
#     XASecondaryModelTrain, 
#     XBSecondaryModelTrain,
    
#     y_fit_train_a_ann,
#     np.array( [get_dense_desc_rank( y_fit_train_a_ann )] ).T,
    
#     y_fit_train_b_ann,
#     np.array( [get_dense_desc_rank( y_fit_train_b_ann )] ).T,

#     np.array([y_fit_train_a_gbm]).T,
#     np.array( [get_dense_desc_rank( y_fit_train_a_gbm )] ).T,

#     np.array([y_fit_train_b_gbm]).T,
#     np.array( [get_dense_desc_rank( y_fit_train_b_gbm )] ).T
# ])


# In[ ]:


# XSecondaryModelTest = np.hstack([
#     XASecondaryModelTest, 
#     XBSecondaryModelTest,
    
#     y_fit_test_a_ann,
#     np.array( [get_dense_desc_rank( y_fit_test_a_ann )] ).T,
    
#     y_fit_test_b_ann,
#     np.array( [get_dense_desc_rank( y_fit_test_b_ann )] ).T,

#     np.array([y_fit_test_a_gbm]).T,
#     np.array( [get_dense_desc_rank( y_fit_test_a_gbm )] ).T,

#     np.array([y_fit_test_b_gbm]).T,
#     np.array( [get_dense_desc_rank( y_fit_test_b_gbm )] ).T
# ])


# In[ ]:


# Write tables to disk
YSecondaryModelTrain.to_parquet(fn_y_train)
YSecondaryModelTest.to_parquet(fn_y_test)

pd.DataFrame(XSecondaryModelTrain).to_parquet(fn_x_train)
pd.DataFrame(XSecondaryModelTest).to_parquet(fn_x_test)


# In[ ]:




