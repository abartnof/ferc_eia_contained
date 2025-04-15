#!/usr/bin/env python
# coding: utf-8

# # Iterate through all tranches using previously-fit 2nd stage model, return tables containing predicted mappings
# 
# __author__: Andrew Bartnof
# 
# __copyright__: Copyright 2025, Rocky Mountain Institute
# 
# __credits__: Alex Engel, Andrew Bartnof

# In[1]:


import pandas as pd
import numpy as np
import os
from glob import glob
import lightgbm as lgb
from tqdm.notebook import tqdm
# from scipy import stats


# In[22]:


data_dir = '/Volumes/Extreme SSD/rematch_eia_ferc1_docker'

dir_x_model_a = os.path.join(data_dir, 'working_data/model_a/model_a_x')
dir_x_model_b = os.path.join(data_dir, 'working_data/model_b/model_b_x')

dir_y_fit_model_a_ann = os.path.join(data_dir, 'working_data/model_a/model_a_ann_y_fit')
dir_y_fit_model_a_gbm = os.path.join(data_dir, 'working_data/model_a/model_a_gbm_y_fit')

dir_y_fit_model_b_ann = os.path.join(data_dir, 'working_data/model_b/model_b_ann_y_fit')
dir_y_fit_model_b_gbm = os.path.join(data_dir, 'working_data/model_b/model_b_gbm_y_fit')

dir_tranches = os.path.join(data_dir, 'working_data/tranches_ferc_to_eia')

dir_y_fit_out = os.path.join(data_dir, 'working_data/model_second_stage/model_z_gbm_y_fit')

fn_model2 = os.path.join(data_dir, 'working_data/model_second_stage/model_second_stage_training/model_2.txt')

fn_top_one_mappings = os.path.join(data_dir, 'output_data/top_one_mappings.parquet')
fn_top_ten_mappings = os.path.join(data_dir, 'output_data/top_ten_mappings.parquet')


# In[3]:


def extract_tranche_id(dir, colname):
    # For any given directory, search for all the applicable parquet files, and
    # return the results as a table with two columns, the parquet files (called whatever
    # you input as colname), and the extracted tranche_id
    # dir = dir_model_a
    ff = glob(dir + '/*.parquet')
    Cte = pd.DataFrame({colname:ff})
    Cte['tranche_id'] = Cte[colname].str.extract('([0-9]{4}_[0-9]{3}(?=\\.parquet))')
    Cte = Cte.set_index('tranche_id', drop=True)
    return Cte

def add_grouped_rank(ID, YFit):
    # For any ID file and YFit file, return a table with y_fit and the ranks for the y_fits
    Cte = ID[['record_id_ferc1']].copy()
    Cte = pd.concat([Cte, YFit], axis=1)
    Cte['y_fit_rank'] = Cte.groupby('record_id_ferc1')['y_fit'].rank(method='dense', ascending=False)
    Cte = Cte[['y_fit', 'y_fit_rank']]
    return Cte


# Load model, locate all filenames and join them together

# In[4]:


mod2 = lgb.Booster(model_file=fn_model2)


# In[5]:


FN = pd.concat([
        extract_tranche_id(dir=dir_tranches, colname='fn_id'),
        extract_tranche_id(dir=dir_x_model_a, colname='fn_x_a'),
        extract_tranche_id(dir=dir_x_model_b, colname='fn_x_b'),
        extract_tranche_id(dir=dir_y_fit_model_a_ann, colname='fn_y_fit_a_ann'),
        extract_tranche_id(dir=dir_y_fit_model_a_gbm, colname='fn_y_fit_a_gbm'),
        extract_tranche_id(dir=dir_y_fit_model_b_ann, colname='fn_y_fit_b_ann'),
        extract_tranche_id(dir=dir_y_fit_model_b_gbm, colname='fn_y_fit_b_gbm')
    ], axis=1, join="outer")

# add location for output
FN['y_fit_out'] = [os.path.join(dir_y_fit_out, 'model_z_gbm_y_fit__' + tranche_id + '.parquet') for tranche_id in FN.index.values]

FN.head(2)


# The input for this model should look like this:
# - X encoding A
# - X encoding B
# - y-fit, y-fit ranks from ANN A
# - y-fit, y-fit ranks from GBM A
# - y-fit, y-fit ranks from ANN B
# - y-fit, y-fit ranks from GBM B

# # Iterate through tranches and get fitted values

# For each row in the FN table, note the top 20 mappings

# In[13]:


top_mapping_dict = {tranche:None for tranche in FN.index.values}

for index, row in tqdm( FN.iterrows(), total=len(FN) ):
    
    ID = pd.read_parquet(row['fn_id'])
    X1A = pd.read_parquet(row['fn_x_a'])
    X1B = pd.read_parquet(row['fn_x_b'])
    YFit1AAnn = pd.read_parquet(row['fn_y_fit_a_ann'])
    YFit1AGbm = pd.read_parquet(row['fn_y_fit_a_gbm'])
    YFit1BAnn = pd.read_parquet(row['fn_y_fit_b_ann'])
    YFit1BGbm = pd.read_parquet(row['fn_y_fit_b_gbm'])
    
    X = np.hstack((
        X1A.values, 
        X1B.values,
        add_grouped_rank(ID=ID, YFit=YFit1AAnn).values,
        add_grouped_rank(ID=ID, YFit=YFit1AGbm).values,
        add_grouped_rank(ID=ID, YFit=YFit1BAnn).values,
        add_grouped_rank(ID=ID, YFit=YFit1BGbm).values
    ))
    
    y_fit2 = mod2.predict(X)
    
    ID = ID
    Cte = ID.copy()
    Cte['y_fit'] = y_fit2
    Cte['y_fit_rank'] = Cte.groupby('record_id_ferc1')['y_fit'].rank(method='dense', ascending=False)
    OutputYFitRank = Cte[['y_fit', 'y_fit_rank']]

    # Write each tranche's y-fit to disk-- then collect top mappings in a dictionary
    OutputYFitRank.to_parquet(path=row['y_fit_out'], index=False) 
    
    TopMappings = Cte.sort_values(['record_id_ferc1', 'y_fit'], ascending=False).groupby('record_id_ferc1').head(20).reset_index(drop=True)    
    top_mapping_dict[index] = TopMappings


# Save two files-- the top match only, and the top 10 matches, jic someone needs to poke around and find some good alternatives

# In[14]:


TopMappings = pd.concat(top_mapping_dict.values(), axis=0).reset_index(drop=True)
TopMappings = TopMappings.sort_values(['record_id_ferc1', 'y_fit', 'record_id_eia'], ascending=[True, False, True])


# In[19]:


TopOne = TopMappings.groupby('record_id_ferc1').head(1).reset_index(drop=True)
TopTen = TopMappings.groupby('record_id_ferc1').head(10).reset_index(drop=True)


# In[23]:


TopOne.to_parquet(fn_top_one_mappings)
TopTen.to_parquet(fn_top_ten_mappings)

