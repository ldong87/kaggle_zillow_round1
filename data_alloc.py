#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 11:16:31 2017

@author: ldong
"""
import numpy as np
import cPickle as pk
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error

with open('bundle_woImpute.pkl','rb') as f:
    train_x, train_y, all_x, flag_all_nan, na_pct_all, train_x_17, train_y_17, all_x_17, flag_all_nan_17 = pk.load(f)
    
train_x.loc[:,'date'] = train_x.loc[:,'date']+12
train_x_17.loc[:,'date'] = train_x_17.loc[:,'date']-12

tmp = train_x
train_x = train_x_17
train_x_17 = tmp

tmp = train_y
train_y = train_y_17
train_y_17 = train_y

tmp = all_x
all_x = all_x_17
all_x_17 = tmp

tmp = flag_all_nan
flag_all_nan = flag_all_nan_17
flag_all_nan_17 = flag_all_nan


def real_mae(pred):
    y17 = pd.read_csv('/workspace/ldong/ml/StackNet/input/train_2017.csv', low_memory=False)#.sort_values('parcelid')
    y16 = pd.read_csv('/workspace/ldong/ml/StackNet/input/train_2016_v2.csv', low_memory=False)#.sort_values('parcelid')
    y = pd.concat([y16,y17], axis=0)
    pred = pred.merge(y,on='parcelid',how='left')
    print 'Raw MAE: ', mean_absolute_error(pred.filter(regex='f_'),pred.logerror)


def write_sub(sub,flag_clip=False):
    sample_sub = pd.read_csv('../../../input/sample_submission.csv',low_memory=False)
    sub[0][flag_all_nan] = np.mean(sub[0])
    sub[1][flag_all_nan] = np.mean(sub[1])
    sub[2][flag_all_nan] = np.mean(sub[2])
    sub[3][flag_all_nan_17] = np.mean(sub[3])
    sub[4][flag_all_nan_17] = np.mean(sub[4])
    sub[5][flag_all_nan_17] = np.mean(sub[5])
    print 'mean logerrors:\n 16_10:%.4f,   16_11:%.4f,   16_12:%.4f \n 17_10:%.4f,   17_11:%.4f,   17_12:%.4f' % \
    (np.mean(sub[0]),np.mean(sub[1]),np.mean(sub[2]),np.mean(sub[3]),np.mean(sub[4]),np.mean(sub[5]))
    if flag_clip:
        output = pd.DataFrame({'ParcelId': sample_sub['ParcelId'].astype(np.int32), 
                               '201610': np.clip(sub[0],-0.08,0.08), 
                               '201611': np.clip(sub[1],-0.08,0.08), 
                               '201612': np.clip(sub[2],-0.08,0.08),
                               '201710': np.clip(sub[3],-0.08,0.08), 
                               '201711': np.clip(sub[4],-0.08,0.08), 
                               '201712': np.clip(sub[5],-0.08,0.08)})
    else:
        output = pd.DataFrame({'ParcelId': sample_sub['ParcelId'].astype(np.int32), 
                           '201610': sub[0], 
                           '201611': sub[1], 
                           '201612': sub[2],
                           '201710': sub[3], 
                           '201711': sub[4], 
                           '201712': sub[5]})
    cols = output.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    output = output[cols]

    output.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False, float_format='%.4f')
        

def vw_format(tr_x, tr_y, flag_all=False):
    tr_vw = []
    if flag_all:
        for i in range(tr_x.shape[0]):
            tmp = np.array2string(tr_x[i], separator='', max_line_width=np.inf).replace('[','').replace(']','')
            tr_vw.append(" | {x}".format(x=tmp))
    else:
        tr_y = tr_y.reset_index(drop=True)
        for i in range(tr_x.shape[0]):
            tmp = np.array2string(tr_x[i], separator='', max_line_width=np.inf).replace('[','').replace(']','')
            tr_vw.append("{label} | {x}".format(label=tr_y.loc[i,'logerror'], x=tmp))
    return tr_vw

def data_vw(tr_x, tr_y, a_x, tr_x17, tr_y17, a_x17):
    pid_all = a_x.loc[:,'parcelid']
    pid_all17 = a_x17.loc[:,'parcelid']
    a_x['parcelid'] = 9
    a_x17['parcelid'] = 21
    a_x = vw_format(a_x.as_matrix(), [], True)
    a_x17 =vw_format(a_x17.as_matrix(), [], True)
    
    tr_x_q1, tr_y_q1, tr_x_q2, tr_y_q2, pid_q2 = data_quarter(tr_x, tr_y, tr_x17, tr_y17, 1, acc=False)
    tr_q1 = vw_format(tr_x_q1.as_matrix(), tr_y_q1)
    tr_q2 = vw_format(tr_x_q2.as_matrix(), tr_y_q2)
    
    _, _, _, _, pid_q3 = data_quarter(tr_x, tr_y, tr_x17, tr_y17, 2, acc=False)
    
    tr_x_q3, tr_y_q3, tr_x_q4, tr_y_q4, pid_q4 = data_quarter(tr_x, tr_y, tr_x17, tr_y17, 3, acc=False)
    tr_q3 = vw_format(tr_x_q3.as_matrix(), tr_y_q3)
    tr_q4 = vw_format(tr_x_q4.as_matrix(), tr_y_q4)
    
    _, _, _, _, pid_q5 = data_quarter(tr_x, tr_y, tr_x17, tr_y17, 4, acc=False)
    
    tr_x_q5, tr_y_q5, tr_x_q6, tr_y_q6, pid_q6 = data_quarter(tr_x, tr_y, tr_x17, tr_y17, 5, acc=False)
    tr_q5 = vw_format(tr_x_q5.as_matrix(), tr_y_q5)
    tr_q6 = vw_format(tr_x_q6.as_matrix(), tr_y_q6)
    
    _, _, tr_x_q7, tr_y_q7, pid_q7 = data_quarter(tr_x, tr_y, tr_x17, tr_y17, 6, acc=False)
    tr_q7 = vw_format(tr_x_q7.as_matrix(), tr_y_q7)
        
    with open('bundleVW_woOutlier.pkl','wb') as f:
        pk.dump([tr_q1, tr_y_q1,
                 tr_q2, tr_y_q2, pid_q2,
                 tr_q3, tr_y_q3, pid_q3,
                 tr_q4, tr_y_q4, pid_q4,
                 tr_q5, tr_y_q5, pid_q5,
                 tr_q6, tr_y_q6, pid_q6,
                 tr_q7, tr_y_q7, pid_q7,
                 a_x, pid_all,
                 a_x17, pid_all17], f, protocol=pk.HIGHEST_PROTOCOL)
    
def data_quarter(tr_x, tr_y, tr_x17, tr_y17, quarter, nextQ=True, acc=True):
    tr_y = tr_y.drop('parcelid', axis=1)
    tr_y17 = tr_y17.drop('parcelid', axis=1)
    if quarter == 1:
        pid = tr_x.loc[(tr_x.date >= quarter*3+1) & (tr_x.date <= (quarter+1)*3),'parcelid']
        tr_x = tr_x.drop('parcelid', axis=1)
        
        x_train = tr_x[tr_x.date <= quarter*3]
        y_train = tr_y[tr_x.date <= quarter*3]
        x_valid = tr_x[(tr_x.date >= quarter*3+1) & (tr_x.date <= (quarter+1)*3)]
        y_valid = tr_y[(tr_x.date >= quarter*3+1) & (tr_x.date <= (quarter+1)*3)]
        
    elif quarter == 2:
        pid = tr_x.loc[(tr_x.date >= quarter*3+1) & (tr_x.date <= (quarter+1)*3),'parcelid']
        tr_x = tr_x.drop('parcelid', axis=1)
        
        if acc:
            x_train = tr_x[(tr_x.date <= quarter*3)] #tr_x[(tr_x.date >= quarter*3-2) & (tr_x.date <= quarter*3)]
            y_train = tr_y[(tr_x.date <= quarter*3)] #tr_y[(tr_x.date >= quarter*3-2) & (tr_x.date <= quarter*3)]
        else:
            x_train = tr_x[(tr_x.date >= quarter*3-2) & (tr_x.date <= quarter*3)]
            y_train = tr_y[(tr_x.date >= quarter*3-2) & (tr_x.date <= quarter*3)]
        
        x_valid = tr_x[(tr_x.date >= quarter*3+1) & (tr_x.date <= (quarter+1)*3)]
        y_valid = tr_y[(tr_x.date >= quarter*3+1) & (tr_x.date <= (quarter+1)*3)]
    elif quarter == 3:
        if nextQ:
            pid = tr_x.loc[(tr_x.date >= quarter*3+1) & (tr_x.date <= (quarter+1)*3),'parcelid']
            tr_x = tr_x.drop('parcelid', axis=1)
            
            if acc:
                x_train = tr_x[(tr_x.date <= quarter*3)] #tr_x[(tr_x.date >= quarter*3-2) & (tr_x.date <= quarter*3)]
                y_train = tr_y[(tr_x.date <= quarter*3)] #tr_y[(tr_x.date >= quarter*3-2) & (tr_x.date <= quarter*3)]
            else:
                x_train = tr_x[(tr_x.date >= quarter*3-2) & (tr_x.date <= quarter*3)]
                y_train = tr_y[(tr_x.date >= quarter*3-2) & (tr_x.date <= quarter*3)]
            
            x_valid = tr_x[(tr_x.date >= quarter*3+1) & (tr_x.date <= (quarter+1)*3)]
            y_valid = tr_y[(tr_x.date >= quarter*3+1) & (tr_x.date <= (quarter+1)*3)]
        else:
            pid = all_x.loc[:,'parcelid'] # all_x is global
            tr_x = tr_x.drop('parcelid', axis=1)
        
            x_train = tr_x[tr_x.date <= quarter*3]
            y_train = tr_y[tr_x.date <= quarter*3]
            x_valid = []
            y_valid = []
    elif quarter == 4:
        pid = tr_x17.loc[(tr_x17.date >= quarter*3+1) & (tr_x17.date <= (quarter+1)*3),'parcelid']
        if (tr_x17.shape[1]-tr_x.shape[1]) == 4:
            tr_x17 = tr_x17.drop(['val_tax_16','val_total_16','val_building_16','val_land_16'], axis=1)
        tr_x17 = tr_x17.drop(['parcelid'], axis=1)
        tr_x = tr_x.drop('parcelid', axis=1)
        
        if acc:
            x_train = tr_x[(tr_x.date <= quarter*3)] #tr_x[(tr_x.date >= quarter*3-2) & (tr_x.date <= quarter*3)]
            y_train = tr_y[(tr_x.date <= quarter*3)] #tr_y[(tr_x.date >= quarter*3-2) & (tr_x.date <= quarter*3)]
        else:
            x_train = tr_x[(tr_x.date >= quarter*3-2) & (tr_x.date <= quarter*3)]
            y_train = tr_y[(tr_x.date >= quarter*3-2) & (tr_x.date <= quarter*3)]
        
        x_valid = tr_x17[(tr_x17.date >= quarter*3+1) & (tr_x17.date <= (quarter+1)*3)]
        y_valid = tr_y17[(tr_x17.date >= quarter*3+1) & (tr_x17.date <= (quarter+1)*3)]
    # 2017 starts here
    elif quarter == 5:
        pid = tr_x17.loc[(tr_x17.date >= quarter*3+1) & (tr_x17.date <= (quarter+1)*3),'parcelid']
        tr_x17 = tr_x17.drop('parcelid', axis=1)
        
        if acc:
            x_train = pd.concat([tr_x.drop('parcelid',axis=1),tr_x17[(tr_x17.date <= quarter*3)]],axis=0) #tr_x17[(tr_x17.date >= quarter*3-2) & (tr_x17.date <= quarter*3)]
            y_train = pd.concat([tr_y,tr_y17[(tr_x17.date <= quarter*3)]],axis=0) #tr_y17[(tr_x17.date >= quarter*3-2) & (tr_x17.date <= quarter*3)]
        else:
            x_train = tr_x17[(tr_x17.date >= quarter*3-2) & (tr_x17.date <= quarter*3)]
            y_train = tr_y17[(tr_x17.date >= quarter*3-2) & (tr_x17.date <= quarter*3)]
       
        x_valid = tr_x17[(tr_x17.date >= quarter*3+1) & (tr_x17.date <= (quarter+1)*3)]
        y_valid = tr_y17[(tr_x17.date >= quarter*3+1) & (tr_x17.date <= (quarter+1)*3)]
    elif quarter == 6:
        pid = tr_x17.loc[(tr_x17.date >= quarter*3+1) & (tr_x17.date <= (quarter+1)*3),'parcelid']
        tr_x17 = tr_x17.drop('parcelid', axis=1)
        
        if acc:
            x_train = pd.concat([tr_x.drop('parcelid',axis=1),tr_x17[(tr_x17.date <= quarter*3)]],axis=0) #tr_x17[(tr_x17.date >= quarter*3-2) & (tr_x17.date <= quarter*3)]
            y_train = pd.concat([tr_y,tr_y17[(tr_x17.date <= quarter*3)]],axis=0) #tr_y17[(tr_x17.date >= quarter*3-2) & (tr_x17.date <= quarter*3)]
        else:
            x_train = tr_x17[(tr_x17.date >= quarter*3-2) & (tr_x17.date <= quarter*3)]
            y_train = tr_y17[(tr_x17.date >= quarter*3-2) & (tr_x17.date <= quarter*3)]
        
        x_valid = tr_x17[(tr_x17.date >= quarter*3+1) & (tr_x17.date <= (quarter+1)*3)]
        y_valid = tr_y17[(tr_x17.date >= quarter*3+1) & (tr_x17.date <= (quarter+1)*3)]
    elif quarter == 7:
        pid = all_x_17.loc[:,'parcelid'] # all_x is global
    
        x_train = pd.concat([tr_x.drop('parcelid',axis=1),tr_x17.drop('parcelid', axis=1)],axis=0)
        y_train = pd.concat([tr_y,tr_y17],axis=0)
        x_valid = []
        y_valid = []    
    return x_train, y_train, x_valid, y_valid, pid