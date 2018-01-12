#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 12:01:49 2017

@author: ldong
"""

import pandas as pd
import cPickle as pk

def feat_filt(tr_x, mon=None):
    feat_good = list(na_pct_all[na_pct_all<=0.5].index)
    if mon in [10,11,12,22,23,24]:
        feat = tr_x.loc[:,feat_good]
        feat.loc[:,'parcelid'] = mon
        feat = feat.rename(columns={'parcelid':'date'})
    else:
        feat_good[0] = 'date'
        feat = tr_x.loc[:,feat_good]
    return feat

def feed_data16():
    with open('featQ23_xgbtree.pkl','rb') as f:
        metaQ2m1,metaQ3m1 = pk.load(f)    
    with open('featQ23_xgblinear.pkl','rb') as f:
        metaQ2m2,metaQ3m2 = pk.load(f)
    with open('featQ23_lgbdt.pkl','rb') as f:
        metaQ2m3,metaQ3m3 = pk.load(f)
    with open('featQ23_xgbdart.pkl','rb') as f:
        metaQ2m4,metaQ3m4 = pk.load(f)    
    with open('featQ23_lgbdart.pkl','rb') as f:
        metaQ2m5,metaQ3m5 = pk.load(f)
    with open('featQ23_vw.pkl','rb') as f:
        metaQ2m6,metaQ3m6 = pk.load(f) 
    with open('featQ23_vwnn.pkl','rb') as f:
        metaQ2m7,metaQ3m7 = pk.load(f) 
    with open('featQ23_catb.pkl','rb') as f:
        metaQ2m8,metaQ3m8 = pk.load(f)
    with open('featQ23_knn.pkl','rb') as f:
        metaQ2m9,metaQ3m9 = pk.load(f) 
    with open('featQ23_lgbrf.pkl','rb') as f:
        metaQ2m10,metaQ3m10 = pk.load(f) 
    with open('featQ23_mlp.pkl','rb') as f:
        metaQ2m11,metaQ3m11 = pk.load(f)
    with open('featQ23_sklelastic.pkl','rb') as f:
        metaQ2m12,metaQ3m12 = pk.load(f) 
        
    metaQ2 = metaQ2m1.join([metaQ2m2.f_xgblinear,metaQ2m3.f_lgbdt,metaQ2m4.f_xgbdart,
                            metaQ2m5.f_lgbdart,  metaQ2m6.f_vw,   metaQ2m7.f_vwnn,
                            metaQ2m8.f_catb,     metaQ2m9.f_knn,  metaQ2m10.f_lgbrf,
                            metaQ2m11.f_mlp,     metaQ2m12.f_sklelastic,
                            feat_filt(tr_x_q2)])
    metaQ3 = metaQ3m1.join([metaQ3m2.f_xgblinear,metaQ3m3.f_lgbdt,metaQ3m4.f_xgbdart,
                            metaQ3m5.f_lgbdart,  metaQ3m6.f_vw,   metaQ3m7.f_vwnn,
                            metaQ3m8.f_catb,     metaQ3m9.f_knn,  metaQ3m10.f_lgbrf,
                            metaQ3m11.f_mlp,     metaQ3m12.f_sklelastic,
                            feat_filt(tr_x_q3)])
        
    with open('featAll16_xgbtree.pkl','rb') as f:
        feat_10m1, feat_11m1, feat_12m1 = pk.load(f)
    with open('featAll16_xgblinear.pkl','rb') as f:
        feat_10m2, feat_11m2, feat_12m2 = pk.load(f)
    with open('featAll16_lgbdt.pkl','rb') as f:
        feat_10m3, feat_11m3, feat_12m3 = pk.load(f)
    with open('featAll16_xgbdart.pkl','rb') as f:
        feat_10m4, feat_11m4, feat_12m4 = pk.load(f)    
    with open('featAll16_lgbdart.pkl','rb') as f:
        feat_10m5, feat_11m5, feat_12m5 = pk.load(f)    
    with open('featAll16_vw.pkl','rb') as f:
        feat_10m6, feat_11m6, feat_12m6 = pk.load(f)    
    with open('featAll16_vwnn.pkl','rb') as f:
        feat_10m7, feat_11m7, feat_12m7 = pk.load(f)    
    with open('featAll16_catb.pkl','rb') as f:
        feat_10m8, feat_11m8, feat_12m8 = pk.load(f)    
    with open('featAll16_knn.pkl','rb') as f:
        feat_10m9, feat_11m9, feat_12m9 = pk.load(f)    
    with open('featAll16_lgbrf.pkl','rb') as f:
        feat_10m10, feat_11m10, feat_12m10 = pk.load(f)    
    with open('featAll16_mlp.pkl','rb') as f:
        feat_10m11, feat_11m11, feat_12m11 = pk.load(f)    
    with open('featAll16_sklelastic.pkl','rb') as f:
        feat_10m12, feat_11m12, feat_12m12 = pk.load(f) 
    feat_10 = feat_10m1.join([feat_10m2.f_xgblinear,feat_10m3.f_lgbdt,feat_10m4.f_xgbdart,
                              feat_10m5.f_lgbdart,  feat_10m6.f_vw,   feat_10m7.f_vwnn,
                              feat_10m8.f_catb,     feat_10m9.f_knn,  feat_10m10.f_lgbrf,
                              feat_10m11.f_mlp,     feat_10m12.f_sklelastic,
                              feat_filt(all_x,10)])
    feat_11 = feat_11m1.join([feat_11m2.f_xgblinear,feat_11m3.f_lgbdt,feat_11m4.f_xgbdart,
                              feat_11m5.f_lgbdart,  feat_11m6.f_vw,   feat_11m7.f_vwnn,
                              feat_11m8.f_catb,     feat_11m9.f_knn,  feat_11m10.f_lgbrf,
                              feat_11m11.f_mlp,     feat_11m12.f_sklelastic,
                              feat_filt(all_x,11)])
    feat_12 = feat_12m1.join([feat_12m2.f_xgblinear,feat_12m3.f_lgbdt,feat_12m4.f_xgbdart,
                              feat_12m5.f_lgbdart,  feat_12m6.f_vw,   feat_12m7.f_vwnn,
                              feat_12m8.f_catb,     feat_12m9.f_knn,  feat_12m10.f_lgbrf,
                              feat_12m11.f_mlp,     feat_12m12.f_sklelastic,
                              feat_filt(all_x,12)])
    return metaQ2, metaQ3, feat_10, feat_11, feat_12
    
def feed_data17():
    with open('featQ4567_xgbtree.pkl','rb') as f:
        metaQ4m1,metaQ5m1,metaQ6m1,metaQ7m1 = pk.load(f)    
    with open('featQ4567_xgblinear.pkl','rb') as f:
        metaQ4m2,metaQ5m2,metaQ6m2,metaQ7m2 = pk.load(f)
    with open('featQ4567_lgbdt.pkl','rb') as f:
        metaQ4m3,metaQ5m3,metaQ6m3,metaQ7m3 = pk.load(f)
    with open('featQ4567_xgbdart.pkl','rb') as f:
        metaQ4m4,metaQ5m4,metaQ6m4,metaQ7m4 = pk.load(f)    
    with open('featQ4567_lgbdart.pkl','rb') as f:
        metaQ4m5,metaQ5m5,metaQ6m5,metaQ7m5 = pk.load(f)    
    with open('featQ4567_vw.pkl','rb') as f:
        metaQ4m6,metaQ5m6,metaQ6m6,metaQ7m6 = pk.load(f)    
    with open('featQ4567_vwnn.pkl','rb') as f:
        metaQ4m7,metaQ5m7,metaQ6m7,metaQ7m7 = pk.load(f)    
    with open('featQ4567_catb.pkl','rb') as f:
        metaQ4m8,metaQ5m8,metaQ6m8,metaQ7m8 = pk.load(f)    
    with open('featQ4567_knn.pkl','rb') as f:
        metaQ4m9,metaQ5m9,metaQ6m9,metaQ7m9 = pk.load(f)    
    with open('featQ4567_lgbrf.pkl','rb') as f:
        metaQ4m10,metaQ5m10,metaQ6m10,metaQ7m10 = pk.load(f)    
    with open('featQ4567_mlp.pkl','rb') as f:
        metaQ4m11,metaQ5m11,metaQ6m11,metaQ7m11 = pk.load(f)    
    with open('featQ4567_sklelastic.pkl','rb') as f:
        metaQ4m12,metaQ5m12,metaQ6m12,metaQ7m12 = pk.load(f)    
        
    metaQ4 = metaQ4m1.join([metaQ4m2.f_xgblinear,metaQ4m3.f_lgbdt,metaQ4m4.f_xgbdart,
                            metaQ4m5.f_lgbdart,  metaQ4m6.f_vw,   metaQ4m7.f_vwnn,
                            metaQ4m8.f_catb,     metaQ4m9.f_knn,  metaQ4m10.f_lgbrf,
                            metaQ4m11.f_mlp,     metaQ4m12.f_sklelastic,
                            feat_filt(tr_x_q4)])
    metaQ5 = metaQ5m1.join([metaQ5m2.f_xgblinear,metaQ5m3.f_lgbdt,metaQ5m4.f_xgbdart,
                            metaQ5m5.f_lgbdart,  metaQ5m6.f_vw,   metaQ5m7.f_vwnn,
                            metaQ5m8.f_catb,     metaQ5m9.f_knn,  metaQ5m10.f_lgbrf,
                            metaQ5m11.f_mlp,     metaQ5m12.f_sklelastic,
                            feat_filt(tr_x_q5)])
    metaQ6 = metaQ6m1.join([metaQ6m2.f_xgblinear,metaQ6m3.f_lgbdt,metaQ6m4.f_xgbdart,
                            metaQ6m5.f_lgbdart,  metaQ6m6.f_vw,   metaQ6m7.f_vwnn,
                            metaQ6m8.f_catb,     metaQ6m9.f_knn,  metaQ6m10.f_lgbrf,
                            metaQ6m11.f_mlp,     metaQ6m12.f_sklelastic,
                            feat_filt(tr_x_q6)])
    metaQ7 = metaQ7m1.join([metaQ7m2.f_xgblinear,metaQ7m3.f_lgbdt,metaQ7m4.f_xgbdart,
                            metaQ7m5.f_lgbdart,  metaQ7m6.f_vw,   metaQ7m7.f_vwnn,
                            metaQ7m8.f_catb,     metaQ7m9.f_knn,  metaQ7m10.f_lgbrf,
                            metaQ7m11.f_mlp,     metaQ7m12.f_sklelastic,
                            feat_filt(tr_x_q7)])
        
    with open('featAll17_xgbtree.pkl','rb') as f:
        feat_22m1, feat_23m1, feat_24m1 = pk.load(f)
    with open('featAll17_xgblinear.pkl','rb') as f:
        feat_22m2, feat_23m2, feat_24m2 = pk.load(f)
    with open('featAll17_lgbdt.pkl','rb') as f:
        feat_22m3, feat_23m3, feat_24m3 = pk.load(f)
    with open('featAll17_xgbdart.pkl','rb') as f:
        feat_22m4, feat_23m4, feat_24m4 = pk.load(f)    
    with open('featAll17_lgbdart.pkl','rb') as f:
        feat_22m5, feat_23m5, feat_24m5 = pk.load(f)    
    with open('featAll17_vw.pkl','rb') as f:
        feat_22m6, feat_23m6, feat_24m6 = pk.load(f)    
    with open('featAll17_vwnn.pkl','rb') as f:
        feat_22m7, feat_23m7, feat_24m7 = pk.load(f)    
    with open('featAll17_catb.pkl','rb') as f:
        feat_22m8, feat_23m8, feat_24m8 = pk.load(f)    
    with open('featAll17_knn.pkl','rb') as f:
        feat_22m9, feat_23m9, feat_24m9 = pk.load(f)    
    with open('featAll17_lgbrf.pkl','rb') as f:
        feat_22m10, feat_23m10, feat_24m10 = pk.load(f)    
    with open('featAll17_mlp.pkl','rb') as f:
        feat_22m11, feat_23m11, feat_24m11 = pk.load(f)    
    with open('featAll17_sklelastic.pkl','rb') as f:
        feat_22m12, feat_23m12, feat_24m12 = pk.load(f)
    feat_22 = feat_22m1.join([feat_22m2.f_xgblinear,feat_22m3.f_lgbdt,feat_22m4.f_xgbdart,
                              feat_22m5.f_lgbdart,  feat_22m6.f_vw,   feat_22m7.f_vwnn,
                              feat_22m8.f_catb,     feat_22m9.f_knn,  feat_22m10.f_lgbrf,
                              feat_22m11.f_mlp,     feat_22m12.f_sklelastic,
                              feat_filt(all_x_17,22)])
    feat_23 = feat_23m1.join([feat_23m2.f_xgblinear,feat_23m3.f_lgbdt,feat_23m4.f_xgbdart,
                              feat_23m5.f_lgbdart,  feat_23m6.f_vw,   feat_23m7.f_vwnn,
                              feat_23m8.f_catb,     feat_23m9.f_knn,  feat_23m10.f_lgbrf,
                              feat_23m11.f_mlp,     feat_23m12.f_sklelastic,
                              feat_filt(all_x_17,23)])
    feat_24 = feat_24m1.join([feat_24m2.f_xgblinear,feat_24m3.f_lgbdt,feat_24m4.f_xgbdart,
                              feat_24m5.f_lgbdart,  feat_24m6.f_vw,   feat_24m7.f_vwnn,
                              feat_24m8.f_catb,     feat_24m9.f_knn,  feat_24m10.f_lgbrf,
                              feat_24m11.f_mlp,     feat_24m12.f_sklelastic,
                              feat_filt(all_x_17,24)])
    return metaQ4, metaQ5, metaQ6, metaQ7, feat_22, feat_23, feat_24