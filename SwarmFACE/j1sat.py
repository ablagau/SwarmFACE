#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 16:43:04 2022

@author: blagau
"""
import numpy as np
import pandas as pd
from viresclient import set_token
from viresclient import SwarmRequest
from .utils import *
from .fac import *
from SwarmFACE.plot_save.single_sat import *

def j1sat(dtime_beg, dtime_end, sat, res='LR', use_filter=True, \
            alpha=None, N3d=None, N2d=None, tincl=None, er_db=0.5, \
            angTHR = 30., savedata=True, saveplot=True):
    '''
    Estimate the FAC density from one Swarm satellite

    Parameters
    ----------
    dtime_beg : str
        Start time
    dtime_end : str
        End time
    sat : [str]
        satellite
    res
    use_filter
    alpha
    N3d
    N2d
    tincl
    er_db
    angTHR
    savedata
    saveplot

    Returns
    -------

    '''

    Bmodel="CHAOS-all='CHAOS-Core'+'CHAOS-Static'+'CHAOS-MMA-Primary'+'CHAOS-MMA-Secondary'"
    request = SwarmRequest()
    request.set_collection("SW_OPER_MAG"+sat[0]+"_"+res+"_1B")
    request.set_products(measurements=["B_NEC"], 
                         auxiliaries=['QDLat','QDLon','MLT'],
                         models=[Bmodel],
                         sampling_step=res_param(res)[0])
    data = request.get_between(start_time = dtime_beg, 
                               end_time = dtime_end,
                               asynchronous=True)   
    print('Used MAG L1B file: ', data.sources[1])
    dat_df = data.as_dataframe()
   
    # checks for missing and bad data points
    # sets bad B_NEC data (zero magnitude in L1b LR files) to NaN. 
    # warns about the data filtering.
    timebads = None
    ndt = (pd.Timestamp(dtime_end) - pd.Timestamp(dtime_beg)).total_seconds()/res_param(res)[1]
    miss_data = 1 if len(dat_df) != ndt else 0
    if miss_data:
         print('MISSING DATA FOR Sw'+sat[0])    
    ind_bads = np.where(\
        np.linalg.norm(np.stack(dat_df['B_NEC'].values), axis = 1)==0)[0]
    if len(ind_bads):
        print('NR. OF BAD DATA POINTS: ', len(ind_bads))
        timebads = dat_df.index[ind_bads]      
        print(timebads.values)
        dat_df = dat_df.drop(dat_df.index[ind_bads])
#        dat_df, timebads = GapsAsNaN(dat_df, ind_bads)

    if miss_data or len(ind_bads):
        print('DATA FILTERING MIGHT NOT WORK PROPERLY')

    ti = dat_df.index
    nti = len(ti)
    # stores position, magnetic field and magnetic model vectors in 
    # corresponding data matrices
    Rsph = dat_df[['Latitude','Longitude','Radius']].values
    Bnec = np.stack(dat_df['B_NEC'].values, axis=0)
    Bmod = np.stack(dat_df['B_NEC_CHAOS-all'].values, axis=0)  
    dBnec = Bnec - Bmod    # magnetic field perturbation in NEC
    # transforms vectors in Gepgraphyc cartesian
    R, MATnec2geo = R_in_GEOC(Rsph)   
    B = np.matmul(MATnec2geo,Bnec[...,None]).reshape(Bnec.shape)
    dB = np.matmul(MATnec2geo,dBnec[...,None]).reshape(dBnec.shape)   
    
    if tincl is not None:
        if isinstance(tincl[0], str): tincl = pd.to_datetime(tincl)   
    # compute the current and stores data in DataFrames
    j_df = singleJfac(ti, R, B, dB, alpha=alpha, N2d=N2d, N3d=N3d, tincl=tincl, \
                   res=res, er_db=er_db, angTHR = angTHR, use_filter = use_filter)    

    dBgeo_df= pd.DataFrame(dB, columns=['dB_xgeo','dB_ygeo','dB_zgeo'],index=ti)
    input_df = dat_df.join(dBgeo_df)     
    
    param = {'dtime_beg':dtime_beg,'dtime_end':dtime_end,'sat': sat, \
             'res': res, 'angTHR': angTHR, 'tincl': tincl, 'Bmodel':Bmodel, \
             'use_filter':use_filter, 'timebads':timebads}
    
    if savedata:
        save_single_sat(j_df, param)
    if saveplot:
        plot_single_sat(j_df, input_df, param)
       
    return j_df, input_df, param 