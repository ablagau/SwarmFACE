#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from viresclient import SwarmRequest
from .fac import *
from .utils import *
from SwarmFACE.plot_save.three_sat import *

def j3sat(dtime_beg, dtime_end, tshift=[0,0,0], use_filter=True,
          er_db=0.5, angTHR = 30.,savedata=True, saveplot=True):
    '''
    High-level routine to estimate the FAC density with three-satellite method

    Parameters
    ----------
    dtime_beg : str
        start time in ISO format 'YYYY-MM-DDThh:mm:ss'
    dtime_end : str
        end time in ISO format
    tshift : [float, float, float]
        array of time shifts (in seconds) to be introduced in satellite data
        in order to achieve a more favorable configuration
    use_filter : boolean
        'True' for data filtering
    er_db : float
        error in magnetic field measurements
    angTHR : float
        minimum accepted angle between the magnetic field vector and
        the spacecraft plane
    savedata : boolean
        'True' for saving the results in an ASCII file
    saveplot : boolean
        'True' for plotting the results

    Returns
    -------
    j_df : DataFrame
        results
    input_df : DataFrame
        input data
    param : dict
        parameters used in the analysis
    '''

    sats=['A','B','C']
    Bmodel="CHAOS-all='CHAOS-Core'+'CHAOS-Static'+'CHAOS-MMA-Primary'+'CHAOS-MMA-Secondary'"    
    dti = pd.date_range(start = dtime_beg, end = dtime_end, freq='s', closed='left')
    ndti = len(dti)
    nsc = len(sats)
    timebads={'sc0':None,'sc1':None,'sc2':None}
    
    ts = np.around(np.array(tshift) - min(tshift))
    tsh3s = np.around(np.mean(ts), decimals=3)
    ndt = ndti - max(ts)
    dt = dti[:ndt].shift(1000.*tsh3s,freq='ms')  # new data timeline
    Rsph, QDref, Bsw, Bmod, dBnec, R, B, dB = (np.full((ndt,nsc,3),np.nan) for i in range(8))    
    
    request = SwarmRequest()
    for sc in range(nsc):
        request.set_collection("SW_OPER_MAG"+sats[sc]+"_LR_1B")
        request.set_products(measurements=["B_NEC"], models=[Bmodel],
                             auxiliaries=['QDLat','QDLon','MLT'],                             
                             sampling_step="PT1S")
        data = request.get_between(start_time = dtime_beg, end_time = dtime_end,
                                   asynchronous=True)   
        print('Used MAG L1B file: ', data.sources[1])
        dat_df = data.as_dataframe()
        # checks for missing and bad data points
        # sets bad B_NEC data (zero magnitude in L1b LR files) to NaN. 
        if len(dat_df) != ndti:
            print('MISSING DATA FOR Sw'+sats[sc])
            sys.exit()
        ind_badsi = np.where(np.linalg.norm(np.stack(dat_df['B_NEC'].values), \
                                           axis = 1)==0)[0]
        if len(ind_badsi):
            dat_df, tbadsi = GapsAsNaN(dat_df, ind_badsi)
            print('NR. OF BAD DATA POINTS FOR Sw'+sats[sc]+': ', len(ind_badsi))
            timebads['sc'+str(sc)] = tbadsi
            print(tbadsi.values)
            print('DATA FILTERING MIGHT NOT WORK PROPERLY')
        
        # Stores position, magnetic field and magnetic model vectors in arrays
        # Takes care of the optional time-shifts introduced between the sensors
        Rsph[:,sc,:] = dat_df[['Latitude','Longitude','Radius']].iloc[ts[sc]:ts[sc]+ndt].values
        QDref[:,sc,:] = dat_df[['QDLat','QDLon','MLT']].iloc[ts[sc]:ts[sc]+ndt].values        
        Bsw[:,sc,:] = np.stack(dat_df['B_NEC'].iloc[ts[sc]:ts[sc]+ndt].values, axis=0)
        Bmod[:,sc,:] = np.stack(dat_df['B_NEC_CHAOS-all'].iloc[ts[sc]:ts[sc]+ndt].values, axis=0)  
        # Computes magnetic field perturbation in NEC
        dBnec[:,sc,:] = Bsw[:,sc,:] - Bmod[:,sc,:]    
        # Computes sats positions (R), magnetic measurements (B), and magnetic  
        # perturbations (dB) in the global geographic (Cartesian) frame. 
        R[:,sc,:], MATnec2geo_sc = R_in_GEOC(np.squeeze(Rsph[:,sc,:]))  
        B[:,sc,:] = np.matmul(MATnec2geo_sc,np.squeeze(Bsw[:,sc,:])[...,None]).reshape(-1,3)
        dB[:,sc,:] = np.matmul(MATnec2geo_sc,np.squeeze(dBnec[:,sc,:])[...,None]).reshape(-1,3)

#     collect all input data in a single DataFrame
    colRsph = pd.MultiIndex.from_product([['Rsph'],sats,['Lat','Lon','Radius']], 
                                       names=['Var','Sat','Com'])
    dfRsph = pd.DataFrame(Rsph.reshape(-1,nsc*3),columns=colRsph,index=dt)

    colQDref = pd.MultiIndex.from_product([['QDref'],sats,['QDLat','QDLon','MLT']], 
                                       names=['Vac','Sat','Com'])
    dfQDref = pd.DataFrame(QDref.reshape(-1,nsc*3),columns=colQDref,index=dt)
    
    colBswBmod = pd.MultiIndex.from_product([['Bsw','Bmod'],sats,['N','E','C']], 
                                       names=['Var','Sat','Com'])
    dfBswBmod = pd.DataFrame(np.concatenate((Bsw.reshape(-1,nsc*3), 
                    Bmod.reshape(-1,nsc*3)),axis=1), columns=colBswBmod,index=dt)

    coldBgeo = pd.MultiIndex.from_product([['dBgeo'],sats,['X','Y','Z']], 
                                       names=['Var','Sat','Com'])
    dfdBgeo = pd.DataFrame(dB.reshape(-1,nsc*3),columns=coldBgeo,index=dt)

    colRgeo = pd.MultiIndex.from_product([['Rgeo'],sats,['X','Y','Z']], 
                                       names=['Var','Sat','Com'])
    dfRgeo = pd.DataFrame(R.reshape(-1,nsc*3),columns=colRgeo,index=dt)

    input_df = pd.concat([dfRsph, dfQDref, dfBswBmod, dfdBgeo, dfRgeo], axis=1)

    j_df = threeJfac(dt, R, B, dB, er_db=er_db, angTHR=angTHR, use_filter=use_filter)

    param = {'dtime_beg':dtime_beg,'dtime_end':dtime_end,'sats': sats, \
             'tshift':ts, 'angTHR': angTHR, 'Bmodel':Bmodel, \
             'use_filter':use_filter, 'timebads':timebads}
    
    if savedata:
        save_three_sat(j_df, param)
    if saveplot:
        plot_three_sat(j_df, input_df, param)        
        
    return j_df, input_df, param