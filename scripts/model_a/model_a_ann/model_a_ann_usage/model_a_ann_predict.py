#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd
import numpy as np
import os

from keras import models, layers, regularizers, optimizers, callbacks, utils, losses, metrics, saving
# from keras.metrics import BinaryAccuracy, AUC, BinaryCrossentropy
from tensorflow.keras.backend import clear_session
from tensorflow import convert_to_tensor
from glob import glob

# from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
# utils.set_random_seed(1)


# In[24]:


# Create a common location for filenames, X and y_fit
data_dir = '/Volumes/Extreme SSD/rematch_eia_ferc1_docker'
dir_x = os.path.join(data_dir, 'working_data/model_a/model_a_x')
dir_y_fit = os.path.join(data_dir, 'working_data/model_a/model_a_ann_y_fit')

fn_list_x = glob(os.path.join(dir_x, '*.parquet'))
FN = pd.DataFrame({'dir_fn_x':fn_list_x})
FN['file_suffix'] = FN['dir_fn_x'].str.extract('(?<=x__)(.*)')

joinme_y_fit = os.path.join(dir_y_fit, 'y_fit__')
FN['dir_fn_y_fit'] = pd.Series([joinme_y_fit + f for f in FN['file_suffix'].values])


# In[25]:


# Load model
fn_model = os.path.join(data_dir, 'working_data/model_a/model_a_training/model_a_ann.keras')
model = saving.load_model(fn_model)


# In[31]:


for i in tqdm(FN.index):
    X = pd.read_parquet(FN['dir_fn_x'][i])
    y_fit = model.predict(X)
    YFit = pd.DataFrame(y_fit).rename(columns={0:'y_fit'})
    YFit.to_parquet(FN['dir_fn_y_fit'][i])


# In[36]:





# In[39]:


get_ipython().system('jupyter nbconvert --to script model_a_ann_predict.ipynb')

