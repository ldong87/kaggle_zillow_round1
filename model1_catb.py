#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 17:59:02 2017

@author: ldong
"""

import numpy as np
import cPickle as pk
import pandas as pd
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.metrics import mean_absolute_error


from data_alloc import *

plt.rcParams['figure.figsize'] = 7, 15

class sklMAE(object):
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        # approxes is list of indexed containers (containers with only __len__ and __getitem__ defined), one container
        # per approx dimension. Each container contains floats.
        # weight is one dimensional indexed container.
        # target is float.
        
        # weight parameter can be None.
        # Returns pair (error, weights sum)
        
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        error_sum = 0.0
        weight_sum = 0.0

        for i in xrange(len(approx)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            error_sum += w * np.abs(target[i] - approx[i] )
        return error_sum, weight_sum

def data_categ(x):
    categ = list(x.filter(regex='categ'))
#    categ.extend(['date','age'])
    categ.extend(list(x.filter(regex='num')))
    categ.extend(list(x.filter(regex='flag')))
    cat_feature = np.squeeze(np.array(np.where(np.in1d(list(x),categ))))
#    for i in xrange(len(categ)):
#        x.loc[:,categ[i]] = x.loc[:,categ[i]].astype('category')
    return cat_feature

def model_run(tr_x, tr_y, tr_x17, tr_y17, quarter, model, pred17=False):
    tr_x, tr_y, valid_x, valid_y, pid = data_quarter(tr_x, tr_y, tr_x17, tr_y17, quarter)
    
    cat_f = data_categ(tr_x)
        
    tr_y = np.squeeze(tr_y.as_matrix())
    valid_y = np.squeeze(valid_y.as_matrix())
    
    model.fit(tr_x, tr_y,cat_features=cat_f, eval_set=[valid_x,valid_y])

#    lgb.plot_importance(bst)
    
    pred = model.predict(valid_x)
    print('valid mae score: {}'.format(mean_absolute_error(valid_y, pred)))
    
    if pred17:
        te_x17 = all_x # all_x is all_x_17 in this case!!!
        te_x17 = te_x17.rename(columns={'parcelid':'date'})
        te_x1 = te_x17
        te_x1.loc[:,'date'] = 7
        
        pred = model.predict(te_x1, thread_count=8)
        
        return pred
    
    pid = pid.to_frame().assign(f_catb=pred)
    return pid

def model_pred(tr_x, tr_y, te_x, tr_x17, tr_y17, te_x17, quarter, model):
    tr_x, tr_y, _, _, pid = data_quarter(tr_x, tr_y, tr_x17, tr_y17, quarter, nextQ=False)
    
    if quarter == 3:
        te_x = te_x.rename(columns={'parcelid':'date'})
        te_x1 = te_x
        te_x1.loc[:,'date'] = 10
        te_x2 = te_x
        te_x2.loc[:,'date'] = 11
        te_x3 = te_x
        te_x3.loc[:,'date'] = 12
        
        cat_f = data_categ(tr_x)
        tr_y = np.squeeze(tr_y.as_matrix())
        
        model.fit(tr_x, tr_y,cat_features=cat_f)        
        pred1 = model.predict(te_x1, thread_count=8)
        pred2 = model.predict(te_x2, thread_count=8)
        pred3 = model.predict(te_x3, thread_count=8)
    elif quarter == 7:
        te_x17 = te_x17.rename(columns={'parcelid':'date'})
        te_x1 = te_x17
        te_x1.loc[:,'date'] = 22
        te_x2 = te_x17
        te_x2.loc[:,'date'] = 23
        te_x3 = te_x17
        te_x3.loc[:,'date'] = 24
        
        cat_f = data_categ(tr_x)
        tr_y = np.squeeze(tr_y.as_matrix())
        
        model.fit(tr_x, tr_y,cat_features=cat_f)        
        pred1 = model.predict(te_x1, thread_count=8)
        pred2 = model.predict(te_x2, thread_count=8)
        pred3 = model.predict(te_x3, thread_count=8)
       
    pred_train = model.predict(tr_x)
    print('train mae score: {}'.format(mean_absolute_error(tr_y, pred_train)))
    pid1 = pid.to_frame().assign(f_catb=pred1)
    pid2 = pid.to_frame().assign(f_catb=pred2)
    pid3 = pid.to_frame().assign(f_catb=pred3)

    return pid1, pid2, pid3


# year 2016
#%%
model = CatBoostRegressor(
    iterations=288, learning_rate=0.01, rsm=1,
    depth=6, l2_leaf_reg=6, bagging_temperature=1,
    loss_function='MAE',
    eval_metric='MAE',
    random_seed=0,
    thread_count=8,
    use_best_model=True,
    od_type='Iter',
    od_wait=100,
    verbose=True)
featQ2 = model_run(train_x, train_y, train_x_17, train_y_17, 1, model)
metaQ2 = featQ2.join(train_y.logerror)
#%%
model = CatBoostRegressor(
    iterations=889, learning_rate=0.01, rsm=1,
    depth=6, l2_leaf_reg=6, bagging_temperature=1,
    loss_function='MAE',
    eval_metric='MAE',
    random_seed=0,
    thread_count=8,
    use_best_model=True,
    od_type='Iter',
    od_wait=100,
    verbose=True)

featQ3 = model_run(train_x, train_y, train_x_17, train_y_17, 2, model)
metaQ3 = featQ3.join(train_y.logerror)
with open('featQ23_catb.pkl','wb') as f:
    pk.dump([metaQ2,metaQ3], f, protocol=pk.HIGHEST_PROTOCOL)
#%% use model trained without 2017 Q3 to predict 
model = CatBoostRegressor(
    iterations=2000, learning_rate=0.01, rsm=0.8,
    depth=6, l2_leaf_reg=6, bagging_temperature=1,
    loss_function='MAE',
    eval_metric='MAE',
    random_seed=0,
    thread_count=8,
    use_best_model=True,
    od_type='Iter',
    od_wait=100,
    verbose=True)

pred_19 = model_run(train_x, train_y, train_x_17, train_y_17, 2, model, True)
with open('catb_pred17_7.pkl','wb') as f:
    pk.dump(pred_19, f, protocol=pk.HIGHEST_PROTOCOL)
# %%
# year 2016
model = CatBoostRegressor(
    iterations=1500, learning_rate=0.01, rsm=1,
    depth=6, l2_leaf_reg=6, bagging_temperature=1,
    loss_function='MAE',
    eval_metric='MAE',
    random_seed=0,
    thread_count=8)

feat_10, feat_11, feat_12 = model_pred(train_x, train_y, all_x, train_x_17, train_y_17, all_x_17, 3, model)

with open('featAll16_catb.pkl','wb') as f:
    pk.dump([feat_10, feat_11, feat_12], f, protocol=pk.HIGHEST_PROTOCOL)
#%% 
write_sub([feat_10.f_catb.as_matrix(),feat_11.f_catb.as_matrix(),feat_12.f_catb.as_matrix(),
           feat_10.f_catb.as_matrix(),feat_11.f_catb.as_matrix(),feat_12.f_catb.as_matrix()])
#%%  
# mae:0.05828486  learning_rate=0.01, rsm=0.8, depth=6, l2_leaf_reg=6, bagging_temperature=1, shrink to 1767
model = CatBoostRegressor(
    iterations=1768, learning_rate=0.01, rsm=0.8,
    depth=6, l2_leaf_reg=6, bagging_temperature=1,
    loss_function='MAE',
    eval_metric='MAE',
    random_seed=0,
    thread_count=8,
    use_best_model=True,
    od_type='Iter',
    od_wait=100,
    verbose=True)

featQ4 = model_run(train_x, train_y, train_x_17, train_y_17, 3, model) 
metaQ4 = featQ4.join(train_y.logerror)

# year 2017
#%%
model = CatBoostRegressor(
    iterations=37, learning_rate=0.01, rsm=0.5,
    depth=10, l2_leaf_reg=3, bagging_temperature=1,
    loss_function='MAE',
    eval_metric='MAE',
    random_seed=0,
    thread_count=8,
    use_best_model=True,
    od_type='Iter',
    od_wait=100,
    verbose=True)
featQ5 = model_run(train_x, train_y, train_x_17, train_y_17, 4, model) # train on 2016Q4, valid on 2017Q1
metaQ5 = featQ5.join(train_y_17.logerror)
#%%
model = CatBoostRegressor(
    iterations=2000, learning_rate=0.01, rsm=0.8,
    depth=6, l2_leaf_reg=6, bagging_temperature=1,
    loss_function='MAE',
    eval_metric='MAE',
    random_seed=0,
    thread_count=8,
    use_best_model=True,
    od_type='Iter',
    od_wait=100,
    verbose=True)

featQ6 = model_run(train_x, train_y, train_x_17, train_y_17, 5, model)
metaQ6 = featQ6.join(train_y_17.logerror)
#real_mae(featQ6)
#%%
model = CatBoostRegressor(
    iterations=2000, learning_rate=0.01, rsm=0.8,
    depth=6, l2_leaf_reg=6, bagging_temperature=1,
    loss_function='MAE',
    eval_metric='MAE',
    random_seed=0,
    thread_count=8,
    use_best_model=True,
    od_type='Iter',
    od_wait=100,
    verbose=True)


featQ7 = model_run(train_x, train_y, train_x_17, train_y_17, 6, model)
metaQ7 = featQ7.join(train_y_17.logerror)
with open('featQ4567_catb.pkl','wb') as f:
    pk.dump([metaQ4,metaQ5,metaQ6,metaQ7], f, protocol=pk.HIGHEST_PROTOCOL)
#%% use model trained without 2017 Q3 to predict 
model = CatBoostRegressor(
    iterations=2000, learning_rate=0.01, rsm=0.8,
    depth=6, l2_leaf_reg=6, bagging_temperature=1,
    loss_function='MAE',
    eval_metric='MAE',
    random_seed=0,
    thread_count=8,
    verbose=True)

pred_19 = model_run(train_x, train_y, train_x_17, train_y_17, 6, model, True)
with open('catb_pred17_7.pkl','wb') as f:
    pk.dump(pred_19, f, protocol=pk.HIGHEST_PROTOCOL)

# year 2017
#%%
model = CatBoostRegressor(
    iterations=1000, learning_rate=0.01, rsm=0.8,
    depth=6, l2_leaf_reg=6, bagging_temperature=1,
    loss_function='MAE',
    eval_metric='MAE',
    random_seed=0,
    thread_count=16,
    verbose=True)

feat_22, feat_23, feat_24 = model_pred(train_x, train_y, all_x, train_x_17, train_y_17, all_x_17, 7, model)

#%%
with open('featAll17_catb.pkl','wb') as f:
    pk.dump([feat_22, feat_23, feat_24], f, protocol=pk.HIGHEST_PROTOCOL)
#%%
write_sub([feat_10.f_catb.as_matrix(),feat_11.f_catb.as_matrix(),feat_12.f_catb.as_matrix(),
           feat_22.f_catb.as_matrix(),feat_23.f_catb.as_matrix(),feat_24.f_catb.as_matrix()])