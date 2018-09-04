# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 00:57:28 2018

@author: ashutosh

Mean Encodings : Multiple Approaches

Dataset - Coursera Kaggle winning competition dataset

Testing mean encodings creation on one of the datasets
"""


path= "D:/DeepLearning/datasets/AML_Specialization_Kaggle/"


## Import Libraries
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import product

## Load Data Files
sales = pd.read_csv(path+"data/sales_train_v2.csv", parse_dates = ['date'])

## Aggregate data to monthly levels instead of date level

index_cols = ['shop_id','item_id','date_block_num']

#For every month we create a grid from all shops/items combination for that month
grid = []
for block_num in sales['date_block_num'].unique():
    cur_shops = sales[sales['date_block_num']==block_num]['shop_id'].unique()
    cur_items = sales[sales['date_block_num']==block_num]['item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops,cur_items,[block_num]])),dtype='int32'))
    
#Turn grid into a dataframe
grid = pd.DataFrame(np.vstack(grid), columns = index_cols, dtype = np.int32)
grid.tail()

#Get aggregated value for (shopid, itemid, month)
gb = sales.groupby(index_cols, as_index=False).agg({'item_cnt_day':{'target':'sum'}})

gb.head()

#fix column names
gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values]
#join aggregated data to the grid
all_data = pd.merge(grid,gb,how='left',on=index_cols).fillna(0)
#sort the data
all_data.sort_values(['date_block_num','shop_id','item_id'],inplace=True)

all_data.tail()