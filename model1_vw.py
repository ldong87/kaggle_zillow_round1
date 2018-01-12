#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 23:08:34 2017

@author: ldong
"""

import numpy as np
import matplotlib.pyplot as plt
from vowpalwabbit import pyvw
from sklearn.metrics import mean_absolute_error
import cPickle as pk
import pandas as pd
import datetime as dt

from data_alloc import *

plt.rcParams['figure.figsize'] = 7, 15

def model_run(tr_x, tr_y, valid_x, valid_y, pid, params):
    param = params[0]
    iter_max = params[1]
    vw = pyvw.vw(*param)
    for iteration in range(iter_max):
        for i in range(len(tr_x)):
            vw.learn(tr_x[i])
    vw.finish()
    vw = pyvw.vw("-i vw.model -t")
    pred = [vw.predict(sample) for sample in valid_x]
    print('valid mae score: {}'.format(mean_absolute_error(valid_y, pred)))
    
    pid = pid.to_frame().assign(f_vw=pred)
    return pid

def model_pred(tr_x, tr_y, te_x, pid, params):
#    tr_x, tr_y, valid_x, valid_y, pid = data_quarter(tr_x, tr_y, tr_x17, tr_y17, quarter, False)
    param = params[0]
    iter_max = params[1]
    vw = pyvw.vw(*param)
    for iteration in range(iter_max):
        for i in range(len(tr_x)):
            vw.learn(tr_x[i])
    vw.finish()
    vw = pyvw.vw("-i vw.model -t")
    pred_tr = [vw.predict(sample) for sample in tr_x]
    print('train mae score: {}'.format(mean_absolute_error(tr_y.logerror, pred_tr)))
    pred = [vw.predict(sample) for sample in te_x]
    pid = pid.to_frame().assign(f_vw=pred)

    return pid



with open('bundleVW_woOutlier.pkl','rb') as f:
    [tr_q1, tr_y_q1,
     tr_q2, tr_y_q2, pid_q2,
     tr_q3, tr_y_q3, pid_q3,
     tr_q4, tr_y_q4, pid_q4,
     tr_q5, tr_y_q5, pid_q5,
     tr_q6, tr_y_q6, pid_q6,
     tr_q7, tr_y_q7, pid_q7,
     a_x, pid_all,
     a_x17, pid_all17] = pk.load(f)

# year 2016
#%%
param = ['-b 6 ' + 
            '--loss_function squared '  + 
            '-l 0.0005 ' + 
            '--l1 0 ' +
            '--l2 0 ' +
            '--holdout_off ' +
            '--total 6 ' +
            '-f vw.model ' +
            '--readable_model vw.readable.model']
iter_max = 3
params = [param, iter_max]

featQ2 = model_run(tr_q1, tr_y_q1, tr_q2, tr_y_q2, pid_q2, params)
metaQ2 = featQ2.join(train_y.logerror)

#%%
param = ['-b 6 ' + 
            '--loss_function squared '  + 
            '-l 0.005 ' + 
            '--l1 0 ' +
            '--l2 0 ' +
            '--holdout_off ' +
            '--nn 100 ' +
            '--dropout ' +
            '--total 6 ' +
            '-f vw.model ' +
            '--readable_model vw.readable.model']
iter_max = 5
params = [param, iter_max]

tr_q12 = []
tr_q12.extend(tr_q1)
tr_q12.extend(tr_q2)
tr_y_q12 = []
tr_y_q12.extend(tr_y_q1)
tr_y_q12.extend(tr_y_q2)
featQ3 = model_run(tr_q12, tr_y_q12, tr_q3, tr_y_q3, pid_q3, params)
metaQ3 = featQ3.join(train_y.logerror)

#%%
with open('featQ23_vw.pkl','wb') as f:
    pk.dump([metaQ2,metaQ3], f, protocol=pk.HIGHEST_PROTOCOL)

# %%
# year 2016
param = ['-b 3 ' + 
            '--loss_function squared '  + 
            '-l 0.005 ' + 
            '--l1 0 ' +
            '--l2 0 ' +
            '--holdout_off ' +
            '--total 6 ' +
            '-f vw.model ' +
            '--readable_model vw.readable.model']
iter_max = 5
params = [param, iter_max]

trainy = pd.concat([tr_y_q1,tr_y_q2,tr_y_q3],axis=0)
train = []
train.extend(tr_q1)
train.extend(tr_q2)
train.extend(tr_q3)
feat_10 = model_pred(train, trainy, a_x, pid_all, params)
#%%
with open('featAll16_vw.pkl','wb') as f:
    pk.dump([feat_10, feat_10, feat_10], f, protocol=pk.HIGHEST_PROTOCOL)
#%%
write_sub([feat_10.f_vw.as_matrix(),feat_10.f_vw.as_matrix(),feat_10.f_vw.as_matrix(),
           feat_10.f_vw.as_matrix(),feat_10.f_vw.as_matrix(),feat_10.f_vw.as_matrix()])
#%%
param = ['-b 5 ' + 
            '--loss_function squared '  + 
            '-l 0.05 ' + 
            '--l1 0 ' +
            '--l2 0 ' +
            '--holdout_off ' +
            '--total 6 ' +
            '-f vw.model ' +
            '--readable_model vw.readable.model']

tr_q123 = []
tr_q123.extend(tr_q12)
tr_q123.extend(tr_q3)
tr_y_q123 = []
tr_y_q123.extend(tr_y_q12)
tr_y_q123.extend(tr_y_q3)
featQ4 = model_run(tr_q123, tr_y_q123, tr_q4, tr_y_q4, pid_q4, params) 
metaQ4 = featQ4.join(train_y.logerror)

# year 2017
#%%
param = ['-b 10 ' + 
            '--loss_function squared '  + 
            '-l 0.005 ' + 
            '--l1 0 ' +
            '--l2 0 ' +
            '--holdout_off ' +
            '--total 6 ' +
            '-f vw.model ' +
            '--readable_model vw.readable.model']

tr_q1234 = []
tr_q1234.extend(tr_q123)
tr_q1234.extend(tr_q4)
tr_y_q1234 = []
tr_y_q1234.extend(tr_y_q123)
tr_y_q1234.extend(tr_y_q4)
featQ5 = model_run(tr_q1234, tr_y_q1234, tr_q5, tr_y_q5, pid_q5, params) # train on 2016Q4, valid on 2017Q1
metaQ5 = featQ5.join(train_y_17.logerror)
#%%
tr_q12345 = []
tr_q12345.extend(tr_q1234)
tr_q12345.extend(tr_q5)
tr_y_q12345 = []
tr_y_q12345.extend(tr_y_q1234)
tr_y_q12345.extend(tr_y_q5)
featQ6 = model_run(tr_q12345, tr_y_q12345, tr_q6, tr_y_q6, pid_q6, params)
metaQ6 = featQ6.join(train_y_17.logerror)
#%%
tr_q123456 = []
tr_q123456.extend(tr_q12345)
tr_q123456.extend(tr_q6)
tr_y_q123456 = []
tr_y_q123456.extend(tr_y_q12345)
tr_y_q123456.extend(tr_y_q6)
featQ7 = model_run(tr_q123456, tr_y_q123456, tr_q7, tr_y_q7, pid_q7, params)
metaQ7 = featQ7.join(train_y_17.logerror)
with open('featQ4567_vw.pkl','wb') as f:
    pk.dump([metaQ4,metaQ5,metaQ6,metaQ7], f, protocol=pk.HIGHEST_PROTOCOL)
    
# year 2017
#%%
trainy17 = pd.concat([trainy,tr_y_q4,tr_y_q5,tr_y_q6,tr_y_q7],axis=0)
train17 = []
train17.extend(train)
train17.extend(tr_q4)
train17.extend(tr_q5)
train17.extend(tr_q6)
train17.extend(tr_q7)
feat_22 = model_pred(train17, trainy17, a_x17, pid_all17, params)
with open('featAll17_vw.pkl','wb') as f:
    pk.dump([feat_22, feat_22, feat_22], f, protocol=pk.HIGHEST_PROTOCOL)
#%%
write_sub([feat_10.f_vw.as_matrix(),feat_10.f_vw.as_matrix(),feat_10.f_vw.as_matrix(),
           feat_22.f_vw.as_matrix(),feat_22.f_vw.as_matrix(),feat_22.f_vw.as_matrix()])