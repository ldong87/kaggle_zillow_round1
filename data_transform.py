#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 15:11:25 2017

@author: ldong
"""

import numpy as np
import cPickle as pk
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os
from sklearn import preprocessing
from geopy.distance import vincenty
from sklearn import neighbors
#from fancyimpute import KNN #SoftImpute, IterativeSVD, BiScaler

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

#train_y = pd.read_csv('train_2016_v2.csv', low_memory=False).set_index('parcelid').sort_values('transactiondate')
train_y = pd.read_csv('../../input/train_2016_v2.csv', low_memory=False)#.sort_values('transactiondate')
#train_y = train_y[~train_y.index.duplicated(keep='first')] # remove multiple solds, keep the latest sold
#train_y.sort_values('parcelid', inplace=True)

all_x = pd.read_csv('../../input/properties_2016.csv', low_memory=False)

#pred_date = ['2016-10-01' for _ in xrange(all_x.shape[0])]
#all_x = pd.concat([pd.DataFrame({'transactiondate':pred_date},index=all_x.index), all_x], axis=1)

train_y_17 = pd.read_csv('../../input/train_2017.csv', low_memory=False)#.sort_values('parcelid')
all_x_17 = pd.read_csv('../../input/properties_2017.csv', low_memory=False)

def property_transform(data_x,pic_name):
  data_x.rename(columns =     {#'transactiondate':'date_trans',
                               'yearbuilt':'date_build_year',
                               'basementsqft':'area_basement',
                               'yardbuildingsqft17':'area_patio',
                               'yardbuildingsqft26':'area_shed', 
                               'poolsizesum':'area_pool',  
                               'lotsizesquarefeet':'area_lot', 
                               'garagetotalsqft':'area_garage',
                               'finishedfloor1squarefeet':'area_firstfloor_finished',
                               'calculatedfinishedsquarefeet':'area_total_calc',
                               'finishedsquarefeet6':'area_base',
                               'finishedsquarefeet12':'area_live_finished',
                               'finishedsquarefeet13':'area_liveperi_finished',
                               'finishedsquarefeet15':'area_total_finished',  
                               'unitcnt':'num_unit',
                               'numberofstories':'num_story',
                               'roomcnt':'num_room',
                               'calculatedbathnbr':'num_50_bath',
                               'bedroomcnt':'num_bedroom',
                               'fullbathcnt':'num_100_bath',
                               'threequarterbathnbr':'num_75_bath',
                               'fireplacecnt':'num_fireplace',
                               'poolcnt':'num_pool',
                               'garagecarcnt':'num_garage',
                               'regionidcounty':'categ_geo_region_cnty',
                               'regionidcity':'categ_geo_region_city',
                               'regionidzip':'categ_geo_region_zip',
                               'regionidneighborhood':'categ_geo_region_neighbor',  
                               'latitude':'geo_lati',
                               'longitude':'geo_longi',
                               'censustractandblock':'geo_census_cnty_trac_bloc',
                               'rawcensustractandblock':'geo_census_cnty_trac_bloc_raw',
                               'taxvaluedollarcnt':'val_total',
                               'structuretaxvaluedollarcnt':'val_building',
                               'landtaxvaluedollarcnt':'val_land',
                               'taxamount':'val_tax',
                               'assessmentyear':'date_tax_year',
                               'taxdelinquencyflag':'flag_tax_delinquency',
                               'taxdelinquencyyear':'date_tax_delinquency_year',
                               'propertyzoningdesc':'categ_zone_desc',
                               'propertylandusetypeid':'categ_zone_type',
                               'propertycountylandusecode':'categ_zone_cnty',
                               'buildingqualitytypeid':'categ_quality',
                               'buildingclasstypeid':'categ_framing',
                               'typeconstructiontypeid':'categ_material',
                               'decktypeid':'categ_deck',
                               'storytypeid':'categ_story',
                               'heatingorsystemtypeid':'categ_heat',
                               'airconditioningtypeid':'categ_aircon',
                               'architecturalstyletypeid':'categ_architec',
                               'hashottuborspa':'flag_tub_spa',
                               'pooltypeid2':'flag_pool_w_spa_tub',
                               'pooltypeid7':'flag_pool_wo_spa_tub'}, inplace=True)
  
  data_x.drop('finishedsquarefeet50', axis=1, inplace=True) # repeat with finishedfloor1squarefeet
  data_x.drop('bathroomcnt', axis=1, inplace=True) # repeat with bathroomcnt
  data_x.drop('pooltypeid10', axis=1, inplace=True) # repeat with hashottuborspa
  data_x.drop('fireplaceflag', axis=1, inplace=True) # redundant due to fireplacecnt
  data_x.drop('num_room', axis=1, inplace=True) # != bath + bed ???
  data_x.drop('date_tax_year', axis=1, inplace=True) # all 2015
  
  data_x['num_pool'].fillna(value=0, inplace=True)
  data_x['num_fireplace'].fillna(value=0, inplace=True)
  data_x['num_75_bath'].fillna(value=0, inplace=True)
  data_x['flag_tub_spa'].fillna(value=0, inplace=True)
  data_x['flag_pool_w_spa_tub'].fillna(value=0, inplace=True)
  data_x['flag_pool_wo_spa_tub'].fillna(value=0, inplace=True)
  data_x['flag_tax_delinquency'].fillna(value=0, inplace=True)
  
  data_x['date_build_year'] = 2018. - data_x['date_build_year']
  data_x.rename(columns={'date_build_year':'age'}, inplace=True)
  
  data_x['flag_tub_spa'] = data_x['flag_tub_spa'].values.astype(float)
  data_x['num_50_bath'] = (data_x['num_50_bath'] - data_x['num_100_bath'])/0.5 
  data_x.loc[data_x['flag_tax_delinquency']=='Y', 'flag_tax_delinquency'] = 1
  
  data_x['geo_lati'] = data_x['geo_lati']#/1000000.0
  data_x['geo_longi'] = data_x['geo_longi']#/1000000.0
  
  data_x['categ_zone_desc'] = data_x['categ_zone_desc'].astype('category').cat.codes
  data_x['categ_zone_type'] = data_x['categ_zone_type'].astype('category').cat.codes
  data_x['categ_zone_cnty'] = data_x['categ_zone_cnty'].astype('category').cat.codes
  data_x['categ_aircon'] = data_x['categ_aircon'].astype('category').cat.codes
  data_x['categ_quality'] = data_x['categ_quality'].astype('category').cat.codes
  data_x['categ_framing'] = data_x['categ_framing'].astype('category').cat.codes
  data_x['categ_story'] = data_x['categ_story'].astype('category').cat.codes
  data_x['categ_architec'] = data_x['categ_architec'].astype('category').cat.codes
  data_x['categ_deck'] = data_x['categ_deck'].astype('category').cat.codes
  data_x['categ_heat'] = data_x['categ_heat'].astype('category').cat.codes
  data_x['categ_material'] = data_x['categ_material'].astype('category').cat.codes
  
  data_x['categ_geo_region_cnty'] = data_x['categ_geo_region_cnty'].astype('category').cat.codes
  data_x['categ_geo_region_city'] = data_x['categ_geo_region_city'].astype('category').cat.codes
  data_x['categ_geo_region_zip'] = data_x['categ_geo_region_zip'].astype('category').cat.codes
  data_x['categ_geo_region_neighbor'] = data_x['categ_geo_region_neighbor'].astype('category').cat.codes

  data_x.fillna(value=-1,inplace=True)
    
  census_cnty_trac_bloc_raw_tmp = data_x.loc[:,'geo_census_cnty_trac_bloc_raw']
    
  census_cnty_trac_bloc_tmp = data_x.loc[:,'geo_census_cnty_trac_bloc']
  census_cnty_trac_bloc_tmp[census_cnty_trac_bloc_tmp==-1] = census_cnty_trac_bloc_raw_tmp[census_cnty_trac_bloc_tmp==-1]*1000000 
  census_cnty_trac_bloc_tmp[census_cnty_trac_bloc_tmp==-1000000] = -1
  census_cnty_trac_bloc_tmp = census_cnty_trac_bloc_tmp.apply(long).apply(str)
  
  data_x['categ_geo_census_cnty'] = pd.Series(census_cnty_trac_bloc_tmp.str[0:4].astype(float), index=data_x.index)
  
  census_trac_tmp = census_cnty_trac_bloc_tmp.str[4:10]
  census_trac_tmp[census_trac_tmp=='']=-1
  data_x['categ_geo_census_trac'] = pd.Series(census_trac_tmp.astype(float), index=data_x.index)
  
  census_bloc_tmp = census_cnty_trac_bloc_tmp.str[10:14] 
  census_bloc_tmp[census_bloc_tmp=='']=-1
  data_x['categ_geo_census_bloc'] = pd.Series(census_bloc_tmp.astype(float), index=data_x.index)
  
  data_x['categ_geo_census_cnty_trac'] = pd.Series(census_cnty_trac_bloc_tmp.str[0:10].astype(float), index=data_x.index)
  
  data_x.drop('geo_census_cnty_trac_bloc_raw', axis=1, inplace=True) # sort-of repeat with censustractandblock
  data_x.drop('fips', axis=1, inplace=True) # repeat with census_cnty
  
  data_x.drop('categ_geo_census_cnty', axis=1, inplace=True) # repeat with categ_georegion_cnty
  data_x.drop('categ_geo_census_trac', axis=1, inplace=True)
  data_x.drop('categ_geo_census_bloc', axis=1, inplace=True)
  data_x.drop('geo_census_cnty_trac_bloc', axis=1, inplace=True)

  data_x['categ_geo_census_cnty_trac'] = data_x['categ_geo_census_cnty_trac'].astype('category').cat.codes
  data_x['categ_geo_census_cnty_trac'] = data_x['categ_geo_census_cnty_trac'] - 1. # cuz fillna before

  # data_x['date_trans'] = data_x['date_trans'].astype('datetime64[ns]')
  # #data_x = pd.concat([pd.DataFrame({'date':all_x['data_trans'].dt.year*100+all_x['date_trans'].dt.month}), data_x], axis=1)
  # data_x = pd.concat([pd.DataFrame({'date': np.ceil(data_x['date_trans'].dt.month.astype(float)/3.) + 4.*(np.ceil(all_x['date_trans'].dt.year.astype(float)/2016.)-1.) }), data_x], axis=1) # use quarter
  # data_x.drop('date_trans', axis=1, inplace=True)
  
  na_pct = (data_x==-1).sum(axis=0) / float(data_x.shape[0])
  
  plt.rcParams['figure.figsize'] = 7, 15
  plt.figure()
  ax = na_pct.sort_values().plot(kind='barh')
  ax.set_xlabel('missing_value_penctage')
  ax.get_figure().savefig('missing_value_pct'+pic_name)
    
  return data_x, na_pct

all_x, na_pct_all = property_transform(all_x, '_all')
all_x_17, na_pct_all_17 = property_transform(all_x_17, '_all_17')

## include 16 tax values in 17
#all_x_17.rename(columns={'val_tax':'val_tax_17',
#                         'val_total':'val_total_17',
#                         'val_building':'val_building_17',
#                         'val_land':'val_land_17'},inplace=True)
#    
#all_x_17 = all_x_17.merge(all_x[['parcelid','val_tax','val_total','val_building','val_land']], on='parcelid', how='left')
#
#all_x_17.rename(columns={'val_tax':'val_tax_16',
#                         'val_total':'val_total_16',
#                         'val_building':'val_building_16',
#                         'val_land':'val_land_16'},inplace=True)
#    
#all_x_17.rename(columns={'val_tax_17':'val_tax',
#                         'val_total_17':'val_total',
#                         'val_building_17':'val_building',
#                         'val_land_17':'val_land'},inplace=True)

#flag_na_pct = na_pct_all < 0.83 #threshold_na
#all_x = all_x[all_x.columns[flag_na_pct]]
#all_x_17 = all_x_17[all_x_17.columns[flag_na_pct]]

#with open('prop_2016_clean.pkl','wb') as f:
#  pk.dump(all_x, f, protocol=pk.HIGHEST_PROTOCOL)

def create_train(train_y, all_x, pic_name):
    train_x = train_y.merge(all_x, on='parcelid', how='left')
    train_x.rename(columns={'transactiondate':'date'}, inplace=True) 
    train_x['date'] = train_x['date'].astype('datetime64[ns]')
    
    plt.rcParams['figure.figsize'] = 7, 7
    plt.figure()
    ax = train_x['date'].groupby([train_x["date"].dt.year, train_x["date"].dt.month]).count().plot(kind="bar")
    ax.get_figure().savefig('train_date'+pic_name)
    
#    train_x = pd.concat([pd.DataFrame({'date_trans': np.ceil(train_x['date'].dt.month.astype(float)/3.) + 4.*(np.ceil(train_x['date'].dt.year.astype(float)/2016.)-1.) }), train_x], axis=1) # use quarter
    train_x = pd.concat([pd.DataFrame({'date_trans': np.ceil(train_x['date'].dt.month) + 12.*(np.ceil(train_x['date'].dt.year.astype(float)/2016.)-1.) }), train_x], axis=1) # use quarter
    train_x.drop('date', axis=1, inplace=True)
    train_x.rename(columns={'date_trans':'date'}, inplace=True)
    na_pct_train = (train_x==-1).sum(axis=0) / float(train_x.shape[0])
    
    
    train_x.drop('logerror', axis=1, inplace=True)
    train_y.drop('transactiondate', axis=1, inplace=True)
#    train_y.drop('parcelid', axis=1, inplace=True)
    
## remove outliers
    fac_std = 5.2
    threshold_logerror = [float(i) for i in [train_y.logerror.mean(axis=0)-fac_std*train_y.logerror.std(axis=0), train_y.logerror.mean(axis=0)+fac_std*train_y.logerror.std(axis=0)] ]
    flag_logerror = (train_y.logerror>threshold_logerror[0]) & (train_y.logerror<threshold_logerror[1])
#    threshold_logerror = [-0.4, 0.418 ]
    flag_logerror = (train_y.logerror>threshold_logerror[0]) & (train_y.logerror<threshold_logerror[1])
    train_x = train_x[flag_logerror]
    train_y = train_y[flag_logerror]
    train_x.reset_index(drop=True,inplace=True)
    train_y.reset_index(drop=True,inplace=True)
# limit outlier values
#    train_y.loc[:,'logerror'] = train_y.logerror.clip(-0.08,0.08)
    
    return train_x, train_y, na_pct_train

train_x, train_y, na_pct_train = create_train(train_y, all_x, '_16')
train_x_17, train_y_17, na_pct_train_17 = create_train(train_y_17, all_x_17, '_17')

#all_x.drop('parcelid', axis=1, inplace=True)
#all_x_17.drop('parcelid', axis=1, inplace=True)
#train_x.drop('parcelid', axis=1, inplace=True)
#train_x_17.drop('parcelid', axis=1, inplace=True)

flag_all_nan = all_x.geo_lati == -1 # flag for properties with no features at all
flag_all_nan_17 = all_x_17.geo_lati == -1 # flag for properties with no features at all


#def rows_filter(flag_train, data_x, train_y, fac_std):
#  print 'original num of rows = ', data_x.shape[0], ' columns = ', data_x.shape[1]
#  
#  threshold_logerror = [float(i) for i in [train_y.mean(axis=0)-fac_std*train_y.std(axis=0), train_y.mean(axis=0)+fac_std*train_y.std(axis=0)] ]
#  flag_logerror = (train_y.logerror>threshold_logerror[0]) & (train_y.logerror<threshold_logerror[1])
#  
##  flag_date_valid = ( (data_x['date'].dt.month == 10) & (data_x['date'].dt.day > 14) ) | ( data_x['date'].dt.month > 10 ) 
#  flag_date_valid = np.squeeze(np.zeros([data_x.shape[0],1], dtype='bool')) # no validation data, use cross validation
#  
#  valid_x = data_x[flag_logerror & flag_date_valid]
#  data_x = data_x[flag_logerror & (~ flag_date_valid)]
#  
#  valid_y = train_y[flag_logerror & flag_date_valid]
#  train_y = train_y[flag_logerror & (~ flag_date_valid)]
#  
#  return valid_x, valid_y, data_x, train_y
#
#valid_x, valid_y, train_x, train_y = rows_filter(1, train_x, train_y, 5.2)
 
#valid_x.drop('parcelid', axis=1, inplace=True)

# make sure the parcelid order
sub = pd.read_csv('../../input/sample_submission.csv')
sub.rename(columns={'ParcelId':'parcelid'}, inplace=True)
sub.drop(['201610','201611','201612','201710','201711','201712'],axis=1,inplace=True)
all_x = sub.merge(all_x,how='left',on='parcelid')
all_x_17 = sub.merge(all_x_17,how='left',on='parcelid')

#sub_pid = list(sub.parcelid)
#all_x = all_x.set_index('parcelid')
#all_x = all_x.reindex(sub_pid)
#all_x['parcelid'] = all_x.index
#all_x = all_x[-1:] + all_x[:-1]
#
#all_x_17 = all_x_17.set_index('parcelid')
#all_x_17 = all_x_17.reindex(sub_pid)
#all_x_17['parcelid'] = all_x_17.index
#all_x_17 = all_x_17[-1:] + all_x_17[:-1]


# !!! remember to take care of those rows with no features!!!
with open('bundle_woImpute.pkl', 'wb') as f:
  pk.dump([train_x, train_y, all_x, flag_all_nan, na_pct_all, train_x_17, train_y_17, all_x_17, flag_all_nan_17], f, protocol=pk.HIGHEST_PROTOCOL)

###  imputation
### simple one
##fill_NaN = preprocessing.Imputer(missing_values=-1, strategy='median', axis=0)
##all_x = pd.DataFrame(fill_NaN.fit_transform(all_x), columns=all_x.columns, index=all_x.index)
## complex ones
##all_x.replace(to_replace=-1,value=np.nan, inplace=True)
##all_x = pd.DataFrame(SoftImpute().complete(all_x), columns=all_x.columns, index=all_x.index)
##all_x = pd.DataFrame(IterativeSVD().complete(all_x), columns=all_x.columns, index=all_x.index)
#
#def knn_impute(x, y_train, flagcateg):
#  flagnan = y_train.isnull().values
#  x_train = x[np.invert(flagnan)]
#  x_train_mean = x_train.mean()
#  x_train_max = x_train.max()
#  x_train_min = x_train.min()
#  x_train = (x_train - x_train_mean) / (x_train_max - x_train_min)
#  y_train = y_train[np.invert(flagnan)]
#  x_test = x[flagnan]
#  x_train = x_train.values.astype(np.float32, copy=False)
#  y_train = y_train.values.astype(np.float32, copy=False)
#  x_test = x_test.values.astype(np.float32, copy=False)
#  n_neighbors = int(np.sqrt(x_train.shape[0]))
#  if flagcateg:
#    knn = neighbors.KNeighborsClassifier(n_neighbors, weights='distance', n_jobs=8)
#  else:
#    knn = neighbors.KNeighborsRegressor(n_neighbors, weights='distance', n_jobs=8)
#  y_test = knn.fit(x_train, y_train).predict(x_test)
#  return y_test
##  return pd.DataFrame(y_test, columns=y_data[flagIncomp].columns, index=y_data[flagIncomp].index)
#
#def impute(na_pct_train, train_x, train_x_17, train_y, train_y_17, flag_train, flag_16):
#    na_pct_train.drop(['parcelid'], axis=0, inplace=True) 
#    
#    if flag_train:
#        if flag_16:
#            all_data = train_x.join(train_y, how='left')
#        else:
#            all_data = train_x.append(train_x_17, ignore_index=True)
#    else:
#        if flag_16:
#            all_data = train_x
#        else:
#            all_data = train_x_17
#    
#    all_data.replace(to_replace=-1,value=np.nan, inplace=True)
#    flagIncomp = list(na_pct_train[na_pct_train>0].index)
#    flagComp = list(na_pct_train[na_pct_train==0].index)
#    
#    for i in xrange(len(flagIncomp)): 
#        print flagIncomp[i]
#        y_train = all_data[flagIncomp[i]]
#        flagnan = y_train.isnull().values
#        if ('categ' in flagIncomp[i]) or ('num' in flagIncomp[i]) or ('flag' in flagIncomp[i]):
#            train_x[flagIncomp[i]].values[flagnan] = knn_impute(all_data[flagComp], y_train, 1)
#        else:
#            train_x[flagIncomp[i]].values[flagnan] = knn_impute(all_data[flagComp], y_train, 0)
#    print 'Oh yeah'
#    return train_x
#
#train_x = impute(na_pct_train, train_x, train_x_17, train_y, train_y_17, 1, 1)
#train_x_17 = impute(na_pct_train_17, train_x, train_x_17, train_y, train_y_17, 1, 0)
#all_x.values[np.invert(flag_all_nan)] = impute(na_pct_train, all_x[np.invert(flag_all_nan)], all_x_17[np.invert(flag_all_nan_17)], train_y, train_y_17, 0, 1)
#all_x_17.values[np.invert(flag_all_nan_17)] = impute(na_pct_train_17, all_x[np.invert(flag_all_nan)], all_x_17[np.invert(flag_all_nan_17)], train_y, train_y_17, 0, 0)
#
#    
##all_x.replace(to_replace=-1, value=np.nan, inplace=True)
##all_x_mean = all_x.mean()
##all_x_max = all_x.max()
##all_x_min = all_x.min() 
##all_x = (all_x - all_x_mean) / (all_x_max - all_x_min)
#
##all_x = pd.DataFrame(KNN(k=int(np.sqrt(all_x.shape[0]))).complete(all_x), columns=all_x.columns, index=all_x.index)
##all_x = all_x * (all_x_max - all_x_min) + all_x_mean



