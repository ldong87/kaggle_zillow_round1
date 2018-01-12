#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 17:59:02 2017

@author: ldong
"""

import numpy as np
import cPickle as pk
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import datetime as dt

from data_alloc import *

plt.rcParams['figure.figsize'] = 7, 15

def data_categ(x):
    categ = list(x.filter(regex='categ'))
#    categ.extend(['date','age'])
    categ.extend(list(x.filter(regex='num')))
    categ.extend(list(x.filter(regex='flag')))
    for i in xrange(len(categ)):
        x.loc[:,categ[i]] = x.loc[:,categ[i]].astype('category')
    return x

def model_run(tr_x, tr_y, tr_x17, tr_y17, quarter, params):
    tr_x, tr_y, valid_x, valid_y, pid = data_quarter(tr_x, tr_y, tr_x17, tr_y17, quarter)
    
    tr_x = data_categ(tr_x)
    valid_x = data_categ(valid_x)
    
    tr_y = tr_y.as_matrix()
    valid_y = valid_y.as_matrix()
    
    d_train = lgb.Dataset(tr_x, label=np.squeeze(tr_y))
    d_valid = lgb.Dataset(valid_x, label=np.squeeze(valid_y))
    
    param = params[0]
    iter_max = params[1]
    verbose_iter = params[2]
    early_stop = params[3]
    
    np.random.seed(0)
    bst = lgb.train(param, d_train, iter_max, [d_train,d_valid], ['train','valid'], early_stopping_rounds=early_stop, verbose_eval=verbose_iter)

#    lgb.plot_importance(bst)
    
    pred = bst.predict(valid_x)
    pid = pid.to_frame().assign(f_lgbrf=pred)
    return pid

def model_pred(tr_x, tr_y, te_x, tr_x17, tr_y17, te_x17, quarter, params):
    tr_x, tr_y, _, _, pid = data_quarter(tr_x, tr_y, tr_x17, tr_y17, quarter, False)
    param = params[0]
    iter_max = params[1]
    verbose_iter = params[2]
    
    if quarter == 3:
        te_x = te_x.rename(columns={'parcelid':'date'})
        te_x1 = te_x
        te_x1.loc[:,'date'] = 10
        te_x2 = te_x
        te_x2.loc[:,'date'] = 11
        te_x3 = te_x
        te_x3.loc[:,'date'] = 12
        
        tr_x = data_categ(tr_x)
        te_x1 = data_categ(te_x1)
        te_x2 = data_categ(te_x2)
        te_x3 = data_categ(te_x3)
        
        d_train = lgb.Dataset(tr_x, label=np.squeeze(tr_y))
        
        np.random.seed(0)
        bst = lgb.train(param, d_train, iter_max, [d_train], ['train'], verbose_eval=verbose_iter)
                
        pred1 = bst.predict(te_x1)
        pred2 = bst.predict(te_x2)
        pred3 = bst.predict(te_x3)
    elif quarter == 7:
        te_x17 = te_x17.rename(columns={'parcelid':'date'})
        te_x1 = te_x17
        te_x1.loc[:,'date'] = 22
        te_x2 = te_x17
        te_x2.loc[:,'date'] = 23
        te_x3 = te_x17
        te_x3.loc[:,'date'] = 24
        
        tr_x = data_categ(tr_x)
        te_x1 = data_categ(te_x1)
        te_x2 = data_categ(te_x2)
        te_x3 = data_categ(te_x3)
        
        d_train = lgb.Dataset(tr_x, label=np.squeeze(tr_y))
        
        np.random.seed(0)
        bst = lgb.train(param, d_train, iter_max, [d_train], ['train'], verbose_eval=verbose_iter)
                
        pred1 = bst.predict(te_x1)
        pred2 = bst.predict(te_x2)
        pred3 = bst.predict(te_x3)
    
    pid1 = pid.to_frame().assign(f_lgbrf=pred1)
    pid2 = pid.to_frame().assign(f_lgbrf=pred2)
    pid3 = pid.to_frame().assign(f_lgbrf=pred3)

    return pid1, pid2, pid3


booster = 'rf'

# year 2016
#%%
param = {}
param['learning_rate'] = 0.001 # shrinkage_rate
param['boosting_type'] = 'rf'
param['objective'] = 'regression'
param['metric'] = 'l1'          # or 'mae'
param['max_depth'] = -1
param['lambda_l1'] = 10
param['lambda_l2'] = 50
param['feature_fraction'] = 0.45
param['bagging_fraction'] = 0.45 # sub_row
param['bagging_freq'] = 20
param['num_leaves'] = 100        # num_leaf
param['min_data_in_leaf'] = 500         # min_data_in_leaf
param['verbose'] = 1
param['feature_fraction_seed'] = 2
param['bagging_seed'] = 1
param['tree_learner'] = 'data'
param['nthread'] = 16
iter_max = 423
early_stop = 100
verbose_iter = 100
params = [param, iter_max, verbose_iter, early_stop]

featQ2 = model_run(train_x, train_y, train_x_17, train_y_17, 1, params)
metaQ2 = featQ2.join(train_y.logerror)
#%%
param = {}
param['learning_rate'] = 0.001 # shrinkage_rate
param['boosting_type'] = 'rf'
param['objective'] = 'regression'
param['metric'] = 'l1'          # or 'mae'
param['max_depth'] = -1
param['lambda_l1'] = 10
param['lambda_l2'] = 10
param['feature_fraction'] = 0.45
param['bagging_fraction'] = 0.45 # sub_row
param['bagging_freq'] = 20
param['num_leaves'] = 32        # num_leaf
#param['min_data_in_leaf'] = 500         # min_data_in_leaf
param['verbose'] = 1
param['tree_learner'] = 'data'
param['nthread'] = 16
iter_max = 242
early_stop = 100
verbose_iter = 100
params = [param, iter_max, verbose_iter, early_stop]

featQ3 = model_run(train_x, train_y, train_x_17, train_y_17, 2, params)
metaQ3 = featQ3.join(train_y.logerror)
with open('featQ23_lgb'+booster+'.pkl','wb') as f:
    pk.dump([metaQ2,metaQ3], f, protocol=pk.HIGHEST_PROTOCOL)
    
# %%
# year 2016
param = {}
param['learning_rate'] = 0.001 # shrinkage_rate
param['boosting_type'] = 'rf'
param['objective'] = 'regression'
param['metric'] = 'l1'          # or 'mae'
param['max_depth'] = -1
param['lambda_l1'] = 10
param['lambda_l2'] = 10
param['feature_fraction'] = 0.45
param['bagging_fraction'] = 0.45 # sub_row
param['bagging_freq'] = 20
param['num_leaves'] = 32        # num_leaf
#param['min_data_in_leaf'] = 500         # min_data_in_leaf
param['verbose'] = 1
param['tree_learner'] = 'data'
param['nthread'] = 16
iter_max = 242
early_stop = 100
verbose_iter = 100
params = [param, iter_max, verbose_iter, early_stop]

feat_10, feat_11, feat_12 = model_pred(train_x, train_y, all_x, train_x_17, train_y_17, all_x_17, 3, params)
#%%
with open('featAll16_lgb'+booster+'.pkl','wb') as f:
    pk.dump([feat_10, feat_11, feat_12], f, protocol=pk.HIGHEST_PROTOCOL)
#%%
write_sub([feat_10.f_lgbrf.as_matrix(),feat_11.f_lgbrf.as_matrix(),feat_12.f_lgbrf.as_matrix(),
           feat_10.f_lgbrf.as_matrix(),feat_11.f_lgbrf.as_matrix(),feat_12.f_lgbrf.as_matrix()])
#%%    
param = {}
param['learning_rate'] = 0.0001 # shrinkage_rate
param['boosting_type'] = 'rf'
param['objective'] = 'regression'
param['metric'] = 'l1'          # or 'mae'
param['max_depth'] = 6
param['lambda_l1'] = 10
param['lambda_l2'] = 1
param['feature_fraction'] = 0.45
param['bagging_fraction'] = 0.45 # sub_row
param['bagging_freq'] = 40
param['num_leaves'] = 32       # num_leaf
#param['min_data_in_leaf'] = 500         # min_data_in_leaf
param['verbose'] = 1
param['tree_learner'] = 'data'
param['nthread'] = 16
iter_max = 6
early_stop = 100
verbose_iter = 100
params = [param, iter_max, verbose_iter, early_stop]

featQ4 = model_run(train_x, train_y, train_x_17, train_y_17, 3, params) 
metaQ4 = featQ4.join(train_y.logerror)
# year 2017
#%%
param = {}
param['learning_rate'] = 0.0001 # shrinkage_rate
param['boosting_type'] = 'rf'
param['objective'] = 'regression'
param['metric'] = 'l1'          # or 'mae'
param['max_depth'] = 6
param['lambda_l1'] = 10
param['lambda_l2'] = 50
param['feature_fraction'] = 0.25
param['bagging_fraction'] = 0.45 # sub_row
param['bagging_freq'] = 30
param['num_leaves'] = 32       # num_leaf
#param['min_data_in_leaf'] = 500         # min_data_in_leaf
param['verbose'] = 1
param['tree_learner'] = 'data'
param['nthread'] = 16
iter_max = 6
early_stop = 100
verbose_iter = 100
params = [param, iter_max, verbose_iter, early_stop]

featQ5 = model_run(train_x, train_y, train_x_17, train_y_17, 4, params) # train on 2016Q4, valid on 2017Q1
metaQ5 = featQ5.join(train_y_17.logerror)
#%%
param = {}
param['learning_rate'] = 0.0001 # shrinkage_rate
param['boosting_type'] = 'rf'
param['objective'] = 'regression'
param['metric'] = 'l1'          # or 'mae'
param['max_depth'] = 6
param['lambda_l1'] = 10
param['lambda_l2'] = 50
param['feature_fraction'] = 0.25
param['bagging_fraction'] = 0.85 # sub_row
param['bagging_freq'] = 30
param['num_leaves'] = 16       # num_leaf
param['min_data_in_leaf'] = 500         # min_data_in_leaf
param['verbose'] = 1
param['tree_learner'] = 'data'
param['nthread'] = 16
iter_max = 6
early_stop = 100
verbose_iter = 100
params = [param, iter_max, verbose_iter, early_stop]

featQ6 = model_run(train_x, train_y, train_x_17, train_y_17, 5, params)
metaQ6 = featQ6.join(train_y_17.logerror)
#%%
param = {}
param['learning_rate'] = 0.0001 # shrinkage_rate
param['boosting_type'] = 'rf'
param['objective'] = 'regression'
param['metric'] = 'l1'          # or 'mae'
param['max_depth'] = 12
param['lambda_l1'] = 10
param['lambda_l2'] = 80
param['feature_fraction'] = 0.25
param['bagging_fraction'] = 0.85 # sub_row
param['bagging_freq'] = 10
param['num_leaves'] = 32       # num_leaf
param['min_data_in_leaf'] = 300         # min_data_in_leaf
param['verbose'] = 1
param['tree_learner'] = 'data'
param['nthread'] = 16
iter_max = 6
early_stop = 100
verbose_iter = 100
params = [param, iter_max, verbose_iter, early_stop]

featQ7 = model_run(train_x, train_y, train_x_17, train_y_17, 6, params)
metaQ7 = featQ7.join(train_y_17.logerror)
with open('featQ4567_lgb'+booster+'.pkl','wb') as f:
    pk.dump([metaQ4,metaQ5,metaQ6,metaQ7], f, protocol=pk.HIGHEST_PROTOCOL)
#%% 
# year 2017
feat_22, feat_23, feat_24 = model_pred(train_x, train_y, all_x, train_x_17, train_y_17, all_x_17, 7, params)
with open('featAll17_lgb'+booster+'.pkl','wb') as f:
    pk.dump([feat_22, feat_23, feat_24], f, protocol=pk.HIGHEST_PROTOCOL)
#%%
write_sub([feat_10.f_lgbrf.as_matrix(),feat_11.f_lgbrf.as_matrix(),feat_12.f_lgbrf.as_matrix(),
           feat_22.f_lgbrf.as_matrix(),feat_23.f_lgbrf.as_matrix(),feat_24.f_lgbrf.as_matrix()])