#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 17:59:02 2017

@author: ldong
"""

import numpy as np
import cPickle as pk
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import datetime as dt

from data_alloc import *

plt.rcParams['figure.figsize'] = 7, 15

def model_run(tr_x, tr_y, tr_x17, tr_y17, quarter, params):
    tr_x, tr_y, valid_x, valid_y, pid = data_quarter(tr_x, tr_y, tr_x17, tr_y17, quarter)
    
    tr_y = tr_y.as_matrix()
    valid_y = valid_y.as_matrix()
    
    xg_train = xgb.DMatrix(tr_x, tr_y)
    xg_test = xgb.DMatrix(valid_x, valid_y)
    watchlist = [(xg_train,'train'), (xg_test,'valid')]
    
    param = params[0]
    param['base_score'] = np.mean(tr_y)
    iter_max = params[1]
    verbose_iter = params[2]
    early_stop = params[3]
    
    bst = xgb.train(params=param, dtrain=xg_train, num_boost_round=iter_max, evals=watchlist, early_stopping_rounds=early_stop, verbose_eval=verbose_iter)
    
#    xgb.plot_importance(bst)
    
    pred = bst.predict(xgb.DMatrix(valid_x))
    pid = pid.to_frame().assign(f_xgbdart=pred)
    return pid

def model_pred(tr_x, tr_y, te_x, tr_x17, tr_y17, te_x17, quarter, params):
    tr_x, tr_y, valid_x, valid_y, pid = data_quarter(tr_x, tr_y, tr_x17, tr_y17, quarter, False)
    tr_y = tr_y.as_matrix()
    
    param = params[0]
    param['base_score'] = np.mean(tr_y)
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
        
        xg_train = xgb.DMatrix(tr_x, tr_y)
        watchlist = [(xg_train,'train')]
        
        bst = xgb.train(params=param, dtrain=xg_train, num_boost_round=iter_max, evals=watchlist, verbose_eval=verbose_iter)
        
#        xgb.plot_importance(bst)
        
        pred1 = bst.predict(xgb.DMatrix(te_x1))
        pred2 = bst.predict(xgb.DMatrix(te_x2))
        pred3 = bst.predict(xgb.DMatrix(te_x3))
    elif quarter == 7:
        te_x17 = te_x17.rename(columns={'parcelid':'date'})
        te_x1 = te_x17
        te_x1.loc[:,'date'] = 22
        te_x2 = te_x17
        te_x2.loc[:,'date'] = 23
        te_x3 = te_x17
        te_x3.loc[:,'date'] = 24
        
        xg_train = xgb.DMatrix(tr_x, tr_y)
        watchlist = [(xg_train,'train')]
        
        bst = xgb.train(params=param, dtrain=xg_train, num_boost_round=iter_max, evals=watchlist, verbose_eval=verbose_iter)
        
#        xgb.plot_importance(bst)
        
        pred1 = bst.predict(xgb.DMatrix(te_x1))
        pred2 = bst.predict(xgb.DMatrix(te_x2))
        pred3 = bst.predict(xgb.DMatrix(te_x3))
    
    pid1 = pid.to_frame().assign(f_xgbdart=pred1)
    pid2 = pid.to_frame().assign(f_xgbdart=pred2)
    pid3 = pid.to_frame().assign(f_xgbdart=pred3)
        
#    feat = pd.DataFrame({'parcelid':all_x['parcelid'], 'f_xgb':pred})
    #feat.to_csv('feat_xgb.csv', index=False, float_format='%.8f')
    return pid1, pid2, pid3




booster = 'dart'

# year 2016
#%%
param = {}
param['booster'] = 'dart'
param['objective'] = 'reg:linear'
param['eval_metric'] = 'mae'
param['eta'] = 0.02
param['gamma'] = 0
param['max_depth'] = 6
param['silent'] = 1 
param['nthread'] = 16
param['subsample'] = 0.8
param['colsample_bytree'] = 0.5
param['colsample_bylevel'] = 1
param['min_child_weight'] = 4
param['alpha'] = 10
param['lambda'] = 1
param['sample_type'] = 'uniform'
param['rate_drop'] = 0.35
param['skip_drop'] = 0.5
iter_max =594
early_stop = 100
verbose_iter = 1
params = [param, iter_max, verbose_iter, early_stop]

featQ2 = model_run(train_x, train_y, train_x_17, train_y_17, 1, params)
metaQ2 = featQ2.join(train_y.logerror)

#%%
aram = {}
param['booster'] = 'dart'
param['objective'] = 'reg:linear'
param['eval_metric'] = 'mae'
param['eta'] = 0.02
param['gamma'] = 0
param['max_depth'] = 6
param['silent'] = 1 
param['nthread'] = 16
param['subsample'] = 0.8
param['colsample_bytree'] = 0.5
param['colsample_bylevel'] = 1
param['min_child_weight'] = 4
param['alpha'] = 10
param['lambda'] = 1
param['sample_type'] = 'uniform'
param['rate_drop'] = 0.25
param['skip_drop'] = 0.5
iter_max = 1068
early_stop = 100
verbose_iter = 100
params = [param, iter_max, verbose_iter, early_stop]

featQ3 = model_run(train_x, train_y, train_x_17, train_y_17, 2, params)
metaQ3 = featQ3.join(train_y.logerror)

#%%
with open('featQ23_xgb'+booster+'.pkl','wb') as f:
    pk.dump([metaQ2,metaQ3], f, protocol=pk.HIGHEST_PROTOCOL)
# %%
# year 2016 pred
aram = {}
param['booster'] = 'dart'
param['objective'] = 'reg:linear'
param['eval_metric'] = 'mae'
param['eta'] = 0.02
param['gamma'] = 0
param['max_depth'] = 6
param['silent'] = 1 
param['nthread'] = 16
param['subsample'] = 0.8
param['colsample_bytree'] = 0.5
param['colsample_bylevel'] = 1
param['min_child_weight'] = 4
param['alpha'] = 10
param['lambda'] = 1
param['sample_type'] = 'uniform'
param['rate_drop'] = 0.25
param['skip_drop'] = 0.5
iter_max = 1350
early_stop = 100
verbose_iter = 100
params = [param, iter_max, verbose_iter, early_stop]

feat_10, feat_11, feat_12 = model_pred(train_x, train_y, all_x, train_x_17, train_y_17, all_x_17, 3, params)
#%%
with open('featAll16_xgb'+booster+'.pkl','wb') as f:
    pk.dump([feat_10, feat_11, feat_12], f, protocol=pk.HIGHEST_PROTOCOL)
#%%
write_sub([feat_10.f_xgbdart.as_matrix(),feat_11.f_xgbdart.as_matrix(),feat_12.f_xgbdart.as_matrix(),
           feat_10.f_xgbdart.as_matrix(),feat_11.f_xgbdart.as_matrix(),feat_12.f_xgbdart.as_matrix()])
    
#%%
param = {}
param['booster'] = 'dart'
param['objective'] = 'reg:linear'
param['eval_metric'] = 'mae'
param['eta'] = 0.02
param['gamma'] = 0
param['max_depth'] = 6
param['silent'] = 1 
param['nthread'] = 8 
param['subsample'] = 0.8
param['colsample_bytree'] = 0.5
param['colsample_bylevel'] = 1
param['min_child_weight'] = 4
param['alpha'] = 10
param['lambda'] = 5
param['sample_type'] = 'uniform'
param['rate_drop'] = 0.25
param['skip_drop'] = 0.5
iter_max = 467
early_stop = 100
verbose_iter = 100
params = [param, iter_max, verbose_iter, early_stop]

featQ4 = model_run(train_x, train_y, train_x_17, train_y_17, 3, params) 
metaQ4 = featQ4.join(train_y.logerror)

## year 2017
#%%
param = {}
param['booster'] = 'dart'
param['objective'] = 'reg:linear'
param['eval_metric'] = 'mae'
param['eta'] = 0.02
param['gamma'] = 0
param['max_depth'] = 6
param['silent'] = 1 
param['nthread'] = 16 
param['subsample'] = 0.8
param['colsample_bytree'] = 0.5
param['colsample_bylevel'] = 1
param['min_child_weight'] = 4
param['alpha'] = 10
param['lambda'] = 5
param['sample_type'] = 'uniform'
param['rate_drop'] = 0.25
param['skip_drop'] = 0.5
iter_max = 120
early_stop = 100
verbose_iter = 100
params = [param, iter_max, verbose_iter, early_stop]

featQ5 = model_run(train_x, train_y, train_x_17, train_y_17, 4, params) 
metaQ5 = featQ5.join(train_y_17.logerror)

#%%
param = {}
param['booster'] = 'dart'
param['objective'] = 'reg:linear'
param['eval_metric'] = 'mae'
param['eta'] = 0.02
param['gamma'] = 0
param['max_depth'] = 6
param['silent'] = 1 
param['nthread'] = 8 
param['subsample'] = 0.8
param['colsample_bytree'] = 0.5
param['colsample_bylevel'] = 1
param['min_child_weight'] = 4
param['alpha'] = 10
param['lambda'] = 5
param['sample_type'] = 'uniform'
param['rate_drop'] = 0.25
param['skip_drop'] = 0.5
iter_max = 397
early_stop = 100
verbose_iter = 100
params = [param, iter_max, verbose_iter, early_stop]

featQ6 = model_run(train_x, train_y, train_x_17, train_y_17, 5, params)
metaQ6 = featQ6.join(train_y_17.logerror)
#%%
param = {}
param['booster'] = 'dart'
param['objective'] = 'reg:linear'
param['eval_metric'] = 'mae'
param['eta'] = 0.02
param['gamma'] = 0
param['max_depth'] = 6
param['silent'] = 1 
param['nthread'] = 8 
param['subsample'] = 0.8
param['colsample_bytree'] = 0.5
param['colsample_bylevel'] = 1
param['min_child_weight'] = 4
param['alpha'] = 10
param['lambda'] = 5
param['sample_type'] = 'uniform'
param['rate_drop'] = 0.25
param['skip_drop'] = 0.5
iter_max = 739
early_stop = 100
verbose_iter = 100
params = [param, iter_max, verbose_iter, early_stop]

featQ7 = model_run(train_x, train_y, train_x_17, train_y_17, 6, params)
metaQ7 = featQ7.join(train_y_17.logerror)

#%%
with open('featQ4567_xgb'+booster+'.pkl','wb') as f:
    pk.dump([metaQ4,metaQ5,metaQ6,metaQ7], f, protocol=pk.HIGHEST_PROTOCOL)
    
#%%    
## year 2017 
param = {}
param['booster'] = 'dart'
param['objective'] = 'reg:linear'
param['eval_metric'] = 'mae'
param['eta'] = 0.02
param['gamma'] = 0
param['max_depth'] = 6
param['silent'] = 1 
param['nthread'] = 8 
param['subsample'] = 0.8
param['colsample_bytree'] = 0.5
param['colsample_bylevel'] = 1
param['min_child_weight'] = 4
param['alpha'] = 10
param['lambda'] = 5
param['sample_type'] = 'uniform'
param['rate_drop'] = 0.25
param['skip_drop'] = 0.5
iter_max = 750
early_stop = 100
verbose_iter = 100
params = [param, iter_max, verbose_iter, early_stop]

feat_22, feat_23, feat_24 = model_pred(train_x, train_y, all_x, train_x_17, train_y_17, all_x_17, 7, params)

with open('featAll17_xgb'+booster+'.pkl','wb') as f:
    pk.dump([feat_22, feat_23, feat_24], f, protocol=pk.HIGHEST_PROTOCOL)
#%%
write_sub([feat_10.f_xgbdart.as_matrix(),feat_11.f_xgbdart.as_matrix(),feat_12.f_xgbdart.as_matrix(),
           feat_22.f_xgbdart.as_matrix(),feat_23.f_xgbdart.as_matrix(),feat_24.f_xgbdart.as_matrix()])