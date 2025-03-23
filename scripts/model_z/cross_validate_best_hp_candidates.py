#!/usr/bin/env python
# coding: utf-8

# Stage 1 hyperparameters:
# - These are a 'fait accompli', and the hyperparameters need only be loaded.
# 
# Stage 2, testable hyperparameters:
# - Load the top n number of possible hyperparameters per iteration.
# 
# Load the mostly feature-engineered stage 1 X and Y files
# 
# For each stage 2 hyperparameter x each fold number:
# - Split the folds into premier_model_fits (x2), secondary_model_train (x2), secondary_model_test (x1)
# - Normalize the premier_model_fits X files
# - Train all four input models on the premier_model_fits files
# - Fit a stage 2 model with the contender stage 2 hyperparameters
# - Test the loss, accuracy, etc

# In[1]:


import pandas as pd
import numpy as np
from glob import glob
import os
import re

from keras import models, layers, regularizers, optimizers, callbacks, utils, losses, metrics
from tensorflow.keras.backend import clear_session
from tensorflow import convert_to_tensor

import lightgbm as lgb

from sklearn.preprocessing import StandardScaler
from sklearn import metrics as sklearn_metrics
from scipy import stats
from tqdm import tqdm


# In[2]:


data_dir = '/Volumes/Extreme SSD/rematch_eia_ferc1_docker'
dir_working_model_a_training = os.path.join(data_dir, 'working_data/model_a/model_a_training')
dir_working_model_b_training = os.path.join(data_dir, 'working_data/model_b/model_b_training')

# fn_out = os.path.join(data_dir, 'working_data/model_second_stage/model_second_stage_training/second_stage_model_cv.csv')
dir_out = os.path.join(data_dir, 'working_data/model_second_stage/model_second_stage_training/cv_results/')


# In[28]:


# Get most promising stage 2 hyperparameters from results of grid search

fn_hp2 = os.path.join(data_dir, 'working_data/model_second_stage/model_second_stage_training/gbm_raytune_2025_03_21/gbm_grid_2025_03_21.csv')
HP2 = pd.read_csv(fn_hp2)
HP2 = HP2.loc[ HP2['rank'] < 10 ]
HP2 = HP2.set_index('rank', drop=True)
HP2 = HP2[['config/verbose', 'config/num_trees', 'config/learning_rate', 'config/min_data_in_leaf', 'config/objective', 'config/early_stopping_round', 'config/metrics']]
HP2 = HP2.rename(columns=lambda x: re.sub('^config/','',x))
HP2 = HP2.merge( pd.DataFrame({'fold_num':np.arange(5)}), how='cross')

hp2_dict = HP2.to_dict(orient='index')

for k in hp2_dict.keys():
    # Metrics
    hp2_dict[k]['metrics'] = ['binary_logloss', 'auc']
    # fn_out
    hp2_dict[k]['fn_out'] = 'num_trees_' + str(hp2_dict[k]['num_trees']) + '__learning_rate_' + str(hp2_dict[k]['learning_rate']) + '__min_data_in_leaf_' + str(hp2_dict[k]['min_data_in_leaf']) + '__fold_num_' + str(hp2_dict[k]['fold_num'])
    hp2_dict[k]['fn_out'] = re.sub(r'\.', 'DOT', hp2_dict[k]['fn_out'])
    hp2_dict[k]['fn_out'] = hp2_dict[k]['fn_out'] + '.csv'
    hp2_dict[k]['fn_out'] = os.path.join(dir_out, hp2_dict[k]['fn_out'])


# In[4]:


fn_x_a = os.path.join(dir_working_model_a_training, 'x.parquet')
fn_y_a = os.path.join(dir_working_model_a_training, 'y.parquet')
fn_id = os.path.join(dir_working_model_a_training, 'id.parquet')

fn_x_b = os.path.join(dir_working_model_b_training, 'x.parquet')
fn_y_b = os.path.join(dir_working_model_b_training, 'y.parquet')


# In[5]:


# Define functions

def np_cleaning(X):
    X = np.clip(X, a_min=-3, a_max=3)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X

def prep_x(fit_scaler, X):
    X = fit_scaler.transform(X)
    X = np_cleaning(X)
    # X = convert_to_tensor(X)
    return(X)

def get_grouped_rank(ID, mask, y_fit):
    # Return an array with each y_fit's rank, in descending order + dense method, grouped 
    # by record_id_ferc1
    CteTrain = ID.loc[mask, ['record_id_ferc1']].copy()
    # CteTrain = ID.loc[is_secondary_model_train, ['record_id_ferc1']].copy()
    CteTrain['y_fit'] = y_fit
    # CteTrain['y_fit'] = y_fit_train_a_gbm
    CteTrain.groupby('record_id_ferc1')['y_fit'].rank(method='dense', ascending=False)
    return CteTrain[['y_fit']]
# Cte[['y_fit']] = y_fit_train

def define_folds(values_for_secondary_model_test):
    # If fold f is reserved for secondary model_test, split the remaining folds
    # half into premier_model_fits, and half into secondary_model_train. 
    # Return all values, as numpy arrays
    fold_values = np.arange(5)
    remaining_fold_values = np.setdiff1d(fold_values, values_for_secondary_model_test)
    np.random.shuffle(remaining_fold_values)
    values_for_premier_model_fits = remaining_fold_values[0:2]
    values_for_secondary_model_train = remaining_fold_values[2:]
    
    print('Values for premier model fits: ', values_for_premier_model_fits)
    print('Values for secondary model train: ', values_for_secondary_model_train)
    print('Values for secondary model test: ', values_for_secondary_model_test)
    
    return values_for_premier_model_fits, values_for_secondary_model_train, np.array(values_for_secondary_model_test)


# In[6]:


def get_boolean_masks_for_folds(ID, values_for_premier_model_fits, values_for_secondary_model_train, values_for_secondary_model_test):
    is_premier_model_fits    = np.isin(element=ID.fold.values, test_elements=values_for_premier_model_fits)
    is_secondary_model_train = np.isin(element=ID.fold.values, test_elements=values_for_secondary_model_train)
    is_secondary_model_test  = np.isin(element=ID.fold.values, test_elements=values_for_secondary_model_test)
    return is_premier_model_fits, is_secondary_model_train, is_secondary_model_test


# In[7]:


def fit_ann(params, X, Y):
    # Can be used by model A or B
    clear_session()
    model_ann = models.Sequential()
    model_ann.add(layers.Dropout(rate=params["dropout_1"]))
    model_ann.add(layers.Dense(units=int(params["relu_1"]), activation='relu'))    
    model_ann.add(layers.Dropout(rate=params["dropout_2"]))
    model_ann.add(layers.Dense(units=int(params["relu_2"]), activation='relu'))   
    model_ann.add(layers.Dense(1, activation='sigmoid'))
    
    model_ann.compile(
        loss=losses.BinaryCrossentropy(),
        metrics=[
            metrics.BinaryCrossentropy(),
            metrics.BinaryAccuracy(), 
            metrics.AUC()
        ]
    )
        
    history_ann = model_ann.fit(
        convert_to_tensor(X), Y, epochs=int(params['epochs']), batch_size=128,
        verbose=0
    ) 
    return model_ann


# In[8]:


def fit_gbm(params, X, Y):
    # Can be used by model A or B
    train_set = lgb.Dataset(X, Y)
    model_gbm = lgb.train(
            params = params,
            train_set=train_set   
        )
    return model_gbm


# In[9]:


def clean_x(X, is_premier_model_fits, is_secondary_model_train, is_secondary_model_test):
    # Scale X files 
    XPremierModelFits    = X.loc[is_premier_model_fits]
    XSecondaryModelTrain = X.loc[is_secondary_model_train]
    XSecondaryModelTest  = X.loc[is_secondary_model_test]
    
    standard_scaler = StandardScaler()
    standard_scaler = standard_scaler.fit(XPremierModelFits)
    
    XPremierModelFits    = prep_x(fit_scaler=standard_scaler, X=XPremierModelFits)
    XSecondaryModelTrain = prep_x(fit_scaler=standard_scaler, X=XSecondaryModelTrain)
    XSecondaryModelTest  = prep_x(fit_scaler=standard_scaler, X=XSecondaryModelTest)
    return XPremierModelFits, XSecondaryModelTrain, XSecondaryModelTest


# # Collect Stage 1 Hyperparameters
# 
# Note that these are a 'fait accompli', and need only be read from the disk

# In[10]:


fn_model_a_ann_hp = os.path.join(data_dir, 'working_data/model_a/model_a_training/model_a_ann_hp.csv')
hp1_a_ann = pd.read_csv(fn_model_a_ann_hp).to_dict(orient='list')
hp1_a_ann = {k:hp1_a_ann[k][0] for k in hp1_a_ann.keys()}


# In[11]:


fn_model_a_gbm_hp = os.path.join(data_dir, 'working_data/model_a/model_a_training/model_a_gbm_hp.csv')
hp1_a_gbm = pd.read_csv(fn_model_a_gbm_hp).to_dict(orient='list')
hp1_a_gbm = {k:hp1_a_gbm[k][0] for k in hp1_a_gbm.keys()}
hp1_a_gbm['metrics'] = ['binary_logloss', 'auc']


# In[12]:


fn_model_b_ann_hp = os.path.join(data_dir, 'working_data/model_b/model_b_training/model_b_ann_hp.csv')
hp1_b_ann = pd.read_csv(fn_model_b_ann_hp).to_dict(orient='list')
hp1_b_ann = {k:hp1_b_ann[k][0] for k in hp1_b_ann.keys()}


# In[13]:


fn_model_b_gbm_hp = os.path.join(data_dir, 'working_data/model_b/model_b_training/model_b_gbm_hp.csv')
hp1_b_gbm = pd.read_csv(fn_model_b_gbm_hp).to_dict(orient='list')
hp1_b_gbm = {k:hp1_b_gbm[k][0] for k in hp1_b_gbm.keys()}
hp1_b_gbm['metrics'] = ['binary_logloss', 'auc']


# # Load data

# In[14]:


X_a = pd.read_parquet(fn_x_a)
X_b = pd.read_parquet(fn_x_b)
Y = pd.read_parquet(fn_y_a)
ID = pd.read_parquet(fn_id)


# # Iterate

# In[55]:


for i in tqdm(hp2_dict.keys()):
    
    params = hp2_dict[i]
    
    does_file_exist =  os.path.isfile( params['fn_out'] )
    if not does_file_exist:
        try:            
            
            # Note which fold we're on, and divvy up the data into test/train bits, based on that
            values_for_premier_model_fits, values_for_secondary_model_train, values_for_secondary_model_test = define_folds( params['fold_num'] )
        
            is_premier_model_fits, is_secondary_model_train, is_secondary_model_test = get_boolean_masks_for_folds(
                ID=ID, 
                values_for_premier_model_fits=values_for_premier_model_fits, 
                values_for_secondary_model_train=values_for_secondary_model_train, 
                values_for_secondary_model_test=values_for_secondary_model_test
            )
        
            # Y, ID
            YPremierModelFits = Y.loc[is_premier_model_fits]
            YSecondaryModelTrain = Y.loc[is_secondary_model_train]
            YSecondaryModelTest = Y.loc[is_secondary_model_test]
            
            IDSecondaryModelTest = ID.loc[is_secondary_model_test]
        
            # XA
            XAPremierModelFits, XASecondaryModelTrain, XASecondaryModelTest = clean_x(
                X=X_a, 
                is_premier_model_fits=is_premier_model_fits, 
                is_secondary_model_train=is_secondary_model_train, 
                is_secondary_model_test=is_secondary_model_test
            )
        
            # XB
            XBPremierModelFits, XBSecondaryModelTrain, XBSecondaryModelTest = clean_x(
                X=X_b, 
                is_premier_model_fits=is_premier_model_fits, 
                is_secondary_model_train=is_secondary_model_train, 
                is_secondary_model_test=is_secondary_model_test
            )
        
            # Fit models for stage 1, get y_fit
            model_a_ann = fit_ann(params=hp1_a_ann, X=XAPremierModelFits, Y=YPremierModelFits)
            model_a_gbm = fit_gbm(params=hp1_a_gbm, X=XAPremierModelFits, Y=YPremierModelFits)
            model_b_ann = fit_ann(params=hp1_b_ann, X=XBPremierModelFits, Y=YPremierModelFits)
            model_b_gbm = fit_gbm(params=hp1_b_gbm, X=XBPremierModelFits, Y=YPremierModelFits)
        
            # ANN, stage 2 Train
            y_fit_train_a_ann = model_a_ann.predict(convert_to_tensor(XASecondaryModelTrain))
            y_fit_train_b_ann = model_b_ann.predict(convert_to_tensor(XBSecondaryModelTrain))
        
            # ANN, stage 2 Test
            y_fit_test_a_ann = model_a_ann.predict(convert_to_tensor(XASecondaryModelTest))
            y_fit_test_b_ann = model_b_ann.predict(convert_to_tensor(XBSecondaryModelTest))
        
            # GBM, stage 2 Train
            y_fit_train_a_gbm = model_a_gbm.predict(XASecondaryModelTrain)
            y_fit_train_b_gbm = model_b_gbm.predict(XBSecondaryModelTrain)
        
            # GBM, stage 2 Test
            y_fit_test_a_gbm = model_a_gbm.predict(XASecondaryModelTest)
            y_fit_test_b_gbm = model_b_gbm.predict(XBSecondaryModelTest)
        
            # Collect the above into something the 2nd stage model can use
            mask=is_secondary_model_train
            XSecondaryModelTrain = np.hstack([
                XASecondaryModelTrain, 
                XBSecondaryModelTrain,
                
                y_fit_train_a_ann,
                get_grouped_rank(ID=ID, mask=mask, y_fit=y_fit_train_a_ann).to_numpy(),
                
                y_fit_train_b_ann,
                get_grouped_rank(ID=ID, mask=mask, y_fit=y_fit_train_b_ann).to_numpy(),
            
            
                np.array([y_fit_train_a_gbm]).T,
                get_grouped_rank(ID=ID, mask=mask, y_fit=y_fit_train_a_gbm).to_numpy(),
            
                np.array([y_fit_train_b_gbm]).T,
                get_grouped_rank(ID=ID, mask=mask, y_fit=y_fit_train_b_gbm).to_numpy()
            ])

            mask=is_secondary_model_test           
            XSecondaryModelTest = np.hstack([
                XASecondaryModelTest, 
                XBSecondaryModelTest,
                
                y_fit_test_a_ann,
                get_grouped_rank(ID=ID, mask=mask, y_fit=y_fit_test_a_ann).to_numpy(),
                
                y_fit_test_b_ann,
                get_grouped_rank(ID=ID, mask=mask, y_fit=y_fit_test_b_ann).to_numpy(),
            
                np.array([y_fit_test_a_gbm]).T,
                get_grouped_rank(ID=ID, mask=mask, y_fit=y_fit_test_a_gbm).to_numpy(),
            
                np.array([y_fit_test_b_gbm]).T,
                get_grouped_rank(ID=ID, mask=mask, y_fit=y_fit_test_b_gbm).to_numpy()
            ])

            XTrain = XSecondaryModelTrain
            XTest = XSecondaryModelTest
            YTrain = YSecondaryModelTrain
            YTest = YSecondaryModelTest
            
            
            # Package in training and testing objects
            train_set = lgb.Dataset(XTrain, YTrain)
            test_set  = lgb.Dataset(XTest,  YTest)
            
            # Model
            gbm = lgb.train(
                params,
                train_set
                # valid_sets=[test_set]    
            )
            y_fit = gbm.predict(XTest)
        
            # Goodness of fit
            Framework = IDSecondaryModelTest[['record_id_ferc1']].copy()
            Framework['y_fit'] = y_fit
            Framework['groupwise_max_y_fit'] = Framework.groupby('record_id_ferc1')['y_fit'].transform('max')
            Framework['y_fit_adj'] = Framework['y_fit'] == Framework['groupwise_max_y_fit']
            
            gof_dict = {
                'precision' : sklearn_metrics.precision_score(YTest.values, Framework['y_fit_adj'].values*1),
                'recall' : sklearn_metrics.recall_score(YTest.values, Framework['y_fit_adj'].values*1),
                'log_loss' : sklearn_metrics.log_loss(YTest.values, y_fit),
                'roc_auc' : sklearn_metrics.roc_auc_score(YTest.values, y_fit)
            }
        
            results = stage_2_param_dict[i] | gof_dict
            pd.DataFrame(results, index=[0]).drop('dir_fn_out', axis=1).to_csv(params['fn_out'], index=False)
            
        except:
            print('CV error, moving to next hyperparameters')

