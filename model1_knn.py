#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 17:59:02 2017

@author: ldong
"""

import numpy as np
import cPickle as pk
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from data_alloc import *

plt.rcParams['figure.figsize'] = 7, 15

def model_run(tr_x, tr_y, tr_x17, tr_y17, quarter):
    tr_x, tr_y, valid_x, valid_y, pid = data_quarter(tr_x, tr_y, tr_x17, tr_y17, quarter)
    
    tr_x = MinMaxScaler().fit_transform(tr_x)
    valid_x = (MinMaxScaler().fit_transform(valid_x))
    
    tr_y = np.squeeze(tr_y.as_matrix())
    valid_y = np.squeeze( valid_y.as_matrix())
    
    np.random.seed(0)
    clf = KNeighborsRegressor(n_neighbors=int(np.sqrt(tr_x.shape[0])), weights='distance', p=1)
    clf.fit(tr_x, tr_y)
    pred = clf.predict(valid_x)
    print('valid mae score: {}'.format(mean_absolute_error(valid_y, pred)))
    pid = pid.to_frame().assign(f_knn=pred)
    return pid

def model_pred(tr_x, tr_y, te_x, tr_x17, tr_y17, te_x17, quarter):
    tr_x, tr_y, _, _, pid = data_quarter(tr_x, tr_y, tr_x17, tr_y17, quarter, False)
    tr_x = MinMaxScaler().fit_transform(tr_x.as_matrix())
    if quarter == 3:
        te_x = te_x.rename(columns={'parcelid':'date'})
        te_x1 = te_x
        te_x1.loc[:,'date'] = 10
        te_x1 = MinMaxScaler().fit_transform(te_x1.as_matrix())
        te_x2 = te_x
        te_x2.loc[:,'date'] = 11
        te_x2 = MinMaxScaler().fit_transform(te_x2.as_matrix())
        te_x3 = te_x
        te_x3.loc[:,'date'] = 12
        te_x3 = MinMaxScaler().fit_transform(te_x3.as_matrix())
        tr_y = np.squeeze(tr_y.as_matrix())
        
        np.random.seed(0)
        clf = KNeighborsRegressor(n_neighbors=int(np.sqrt(tr_x.shape[0])), weights='distance', p=1)
        clf.fit(tr_x, tr_y)
        pred1 = clf.predict(te_x1)
        pred2 = clf.predict(te_x2)
        pred3 = clf.predict(te_x3)
    elif quarter == 7:
        te_x17 = te_x17.rename(columns={'parcelid':'date'})
        te_x1 = te_x17
        te_x1.loc[:,'date'] = 22
        te_x1 = MinMaxScaler().fit_transform(te_x1.as_matrix())
        te_x2 = te_x17
        te_x2.loc[:,'date'] = 23
        te_x2 = MinMaxScaler().fit_transform(te_x2.as_matrix())
        te_x3 = te_x17
        te_x3.loc[:,'date'] = 24
        te_x3 = MinMaxScaler().fit_transform(te_x3.as_matrix())
        tr_y = np.squeeze(tr_y.as_matrix())
        
        np.random.seed(0)
        clf = KNeighborsRegressor(n_neighbors=int(np.sqrt(tr_x.shape[0])), weights='distance', p=1)
        print 'training all done!'
        clf.fit(tr_x, tr_y)
        pred1 = clf.predict(te_x1)
        pred2 = pred1 #clf.predict(te_x2)
        pred3 = pred1 #clf.predict(te_x3)
        
    pred_train = clf.predict(tr_x)
    print('train mae score: {}'.format(mean_absolute_error(tr_y, pred_train)))
    pid1 = pid.to_frame().assign(f_knn=pred1)
    pid2 = pid.to_frame().assign(f_knn=pred2)
    pid3 = pid.to_frame().assign(f_knn=pred3)

    return pid1, pid2, pid3



## year 2016
##%%
#featQ2 = model_run(train_x, train_y, train_x_17, train_y_17, 1)
#metaQ2 = featQ2.join(train_y.logerror)
##%%
#featQ3 = model_run(train_x, train_y, train_x_17, train_y_17, 2)
#metaQ3 = featQ3.join(train_y.logerror)
#with open('featQ23_knn.pkl','wb') as f:
#    pk.dump([metaQ2,metaQ3], f, protocol=pk.HIGHEST_PROTOCOL)
#    
## %%
## year 2016
#feat_10, feat_11, feat_12 = model_pred(train_x, train_y, all_x, train_x_17, train_y_17, all_x_17, 3)
#
##%%
#with open('featAll16_knn.pkl','wb') as f:
#    pk.dump([feat_10, feat_11, feat_12], f, protocol=pk.HIGHEST_PROTOCOL)
#    
#write_sub([feat_10.f_knn.as_matrix(),feat_11.f_knn.as_matrix(),feat_12.f_knn.as_matrix(),
#           feat_10.f_knn.as_matrix(),feat_11.f_knn.as_matrix(),feat_12.f_knn.as_matrix()])
##%% 
#featQ4 = model_run(train_x, train_y, train_x_17, train_y_17, 3) 
#metaQ4 = featQ4.join(train_y.logerror)
## year 2017
##%%
#featQ5 = model_run(train_x, train_y, train_x_17, train_y_17, 4) # train on 2016Q4, valid on 2017Q1
#metaQ5 = featQ5.join(train_y_17.logerror)
##%%
#featQ6 = model_run(train_x, train_y, train_x_17, train_y_17, 5)
#metaQ6 = featQ6.join(train_y_17.logerror)
##%%
#featQ7 = model_run(train_x, train_y, train_x_17, train_y_17, 6)
#metaQ7 = featQ7.join(train_y_17.logerror)
#with open('featQ4567_knn.pkl','wb') as f:
#    pk.dump([metaQ4,metaQ5,metaQ6,metaQ7], f, protocol=pk.HIGHEST_PROTOCOL)
    
# year 2017 
#%%
feat_22, feat_23, feat_24 = model_pred(train_x, train_y, all_x, train_x_17, train_y_17, all_x_17, 7)

#%%
with open('featAll17_knn.pkl','wb') as f:
    pk.dump([feat_22, feat_23, feat_24], f, protocol=pk.HIGHEST_PROTOCOL)
    
write_sub([feat_10.f_knn.as_matrix(),feat_11.f_knn.as_matrix(),feat_12.f_knn.as_matrix(),
           feat_22.f_knn.as_matrix(),feat_23.f_knn.as_matrix(),feat_24.f_knn.as_matrix()])