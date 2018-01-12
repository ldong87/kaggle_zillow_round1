#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 14:54:41 2017

@author: ldong
"""

import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def ProjectOnMedian(data1, data2, columnName):
    grpOutcomes = data1.groupby(list([columnName]))['y'].median().reset_index()
    grpCount = data1.groupby(list([columnName]))['y'].count().reset_index()
    grpOutcomes['cnt'] = grpCount.y
    grpOutcomes.drop('cnt', inplace=True, axis=1)
    outcomes = data2['y'].values
    x = pd.merge(data2[[columnName, 'y']], grpOutcomes,
                 suffixes=('x_', ''),
                 how='left',
                 on=list([columnName]),
                 left_index=True)['y']

    
    return x.values

directory = '../../../input/'
tr = pd.read_csv(directory+'train_2016_v2.csv', low_memory=False)
tr17 = pd.read_csv(directory+'train_2017.csv', low_memory=False)
samp = pd.read_csv(directory+'sample_submission.csv', low_memory=False)
prop = pd.read_csv(directory+'properties_2016.csv', low_memory=False)
prop17 = pd.read_csv(directory+'properties_2017.csv', low_memory=False)
## make sure parcelid order
#sub = pd.read_csv('../../../input/sample_submission.csv')
#sub.rename(columns={'ParcelId':'parcelid'}, inplace=True)
#sub_pid = list(sub.parcelid)
#properties.set_index('parcelid')
#properties.reindex(sub_pid)
#properties.loc[:,'parcelid'] = sub.parcelid
#properties17.set_index('parcelid')
#properties17.reindex(sub_pid)
#properties17.loc[:,'parcelid'] = sub.parcelid

#%%
def data_proc(train, properties, sample):
    properties.hashottuborspa = properties.hashottuborspa.astype(str)
    properties.fireplaceflag = properties.fireplaceflag.astype(str)
    for c in properties.columns:
        if properties[c].dtype == 'object':
            print(c)
            lbl = LabelEncoder()
            lbl.fit(list(properties[c].values))
            properties[c] = lbl.transform(list(properties[c].values))
            
    highcardinality = ['airconditioningtypeid',
                       'architecturalstyletypeid',
                       'buildingclasstypeid',
                       'buildingqualitytypeid',
                       'decktypeid',
                       'fips',
                       'hashottuborspa',
                       'heatingorsystemtypeid',
                       'pooltypeid10',
                       'pooltypeid2',
                       'pooltypeid7',
                       'propertycountylandusecode',
                       'propertylandusetypeid',
                       'regionidcity',
                       'regionidcounty',
                       'regionidneighborhood',
                       'regionidzip',
                       'storytypeid',
                       'typeconstructiontypeid',
                       'fireplaceflag',
                       'taxdelinquencyflag']
    
    sample = sample.rename(columns={'ParcelId':'parcelid'})
    sample.head()
    
    train = train.merge(properties, how='left', on='parcelid')
    test = sample.merge(properties, how='left', on='parcelid')
    train['month'] = pd.DatetimeIndex(train['transactiondate']).month
    test['month'] = -1
    logerrors = train.logerror.ravel()
    train.drop(['logerror','transactiondate'],inplace=True,axis=1)
    
    
    test = test[train.columns]
    train.insert(1,'nans',train.isnull().sum(axis=1))
    test.insert(1,'nans',train.isnull().sum(axis=1))
    train['y'] = logerrors
    test['y'] = np.nan
    
    from sklearn.model_selection import KFold
    blindloodata = None
    folds = 20
    kf = KFold(n_splits=folds,shuffle=True,random_state=42)
    for i, (train_index, test_index) in enumerate(kf.split(range(train.shape[0]))):
        print('Fold:',i)
        blindtrain = train.loc[test_index].copy() 
        vistrain = train.loc[train_index].copy()
    
    
    
        for c in highcardinality:
            blindtrain.insert(1,'loo'+c, ProjectOnMedian(vistrain,
                                                         blindtrain,c))
        if(blindloodata is None):
            blindloodata = blindtrain.copy()
        else:
            blindloodata = pd.concat([blindloodata,blindtrain])
    
    for c in highcardinality:
        test.insert(1,'loo'+c, ProjectOnMedian(train,
                                               test,c))
    test.drop(highcardinality,inplace=True,axis=1)
    
    train = blindloodata
    train.drop(highcardinality,inplace=True,axis=1)
    
    feats = train.columns[1:-1]
    for c in feats:
        train[c] = train[c].fillna(train[c].median())
        test[c] = test[c].fillna(train[c].median())
    
    return train, test

#%%
train, test =  data_proc(tr, prop, samp)
train17, test17 =  data_proc(tr17, prop17, samp)
train17.loc[:,'month'] = train17.loc[:,'month']+12

train = pd.concat([train,train17], axis=0).reset_index(drop=True)
test = test17

xtrain = train.loc[(train.y>-0.418)&(train.y< 0.418)].copy()
xtrain = xtrain.reset_index(drop=True)
features = xtrain.columns[1:-1]
xtrain.y = xtrain.y+.418
xtrain.y /= (2*.418)

valid = xtrain.loc[(train.month>=19)&(train.month<=21)]
valid.y = valid.y*2*0.418-0.418
xtrain = xtrain.loc[train.month<=18]

ss = StandardScaler()
ss.fit(pd.concat([xtrain[features],test[features]]))
xtrain[features] = ss.transform(xtrain[features])
valid[features] = ss.transform(valid[features])
test[features] = ss.transform(test[features])

from gp_feat import *
from sklearn.metrics import mean_absolute_error

#validpreds = GP(valid)
#print 'valid mae: %.6f'%mean_absolute_error(validpreds,valid.y) 
testpreds = GP(test)


#%%
sub = pd.read_csv(directory+'sample_submission.csv')
for i, c in enumerate(sub.columns[sub.columns != 'ParcelId']):
    sub[c] = testpreds
sub.to_csv('gp17.csv',index=False,float_format='%.4f')
