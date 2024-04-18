#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy import signal
from viresclient import SwarmRequest
from .utils import *
from .fac import *
from SwarmFACE.plot_save.saveplot_qi import *

def fac_qi(dtime_beg, dtime_end, swB=False, jTHR=0.05, saveplot=False):
    '''
    High-level routine to automatically estimate the quality indices of
    FAC structures

    Parameters
    ----------
    dtime_beg : str
        start time in ISO format 'YYYY-MM-DDThh:mm:ss'
    dtime_end : str
        end time in ISO format
    swB : boolean
        'True' to  the MVA based quality indices for Swarm B
    jTHR : float
        threshold to neglect small FAC densities
    saveplot : boolean
        'True' for plotting the results

    Returns
    -------
    input_df : list
        list of DataFrames (one per satellite per quarter-orbit
        section) with input data
    RBdBAng_df : list
         list of DataFrrames (one per satellite per quarter-orbit
         section) with intermediate variables
    fac_df : list
        list of DataFrrames (one per satellite per quarter-orbit
        section) with estimated single=satellite FAC data
    qimva_df : DataFrame
        MVA results, FAC planarity, and FAC inclination
    qicc_df : DataFrame
        results from the correlation analysis between the magnetic
        field perturbations at the lower Swarm satellites
    param : dict
        parameters used in the analysis
    '''

    fs = 1              # Sampling frequency
    fc = 1./20.         # Cut-off frequency of a 20 s low-pass filter
    w = fc / (fs / 2)   # Normalize the frequency
    butter_ord = 5
    bf, af = signal.butter(butter_ord, w, 'low')
    
    Bmodel="CHAOS-all='CHAOS-Core'+'CHAOS-Static'+'CHAOS-MMA-Primary'+'CHAOS-MMA-Secondary'"
    request = SwarmRequest()

    sats=['A','C']
    if swB:
        sats = ['A','B','C']
    nsc = len(sats)
    
    # Retieves the L1b magnetic field data from the ESA database 
    orbs = np.full((nsc,2),np.nan)
    Bnec_df, tlarges = ([] for i in range(2))
    for sc in range(nsc):
        # find the orbits that contain the analysis interval
        orb1 = request.get_orbit_number(sats[sc], dtime_beg, mission='Swarm')
        orb2 = request.get_orbit_number(sats[sc], dtime_end, mission='Swarm')
        print('sat: sw'+sats[sc]+', orbit start/end: '+str(orb1)+' / '+ 
              str(orb2)+ ',   nr. orbits: '+ str(orb2 - orb1 +1))
        orbs[sc, :] = [orb1, orb2]              
        large_beg, large_end = request.get_times_for_orbits(orb1, orb2, 
                                         mission='Swarm', spacecraft=sats[sc])
        tlarges.append([large_beg, large_end])
        dti = pd.date_range(start= large_beg, end= large_end, freq='s', closed='left')
        # get get B NEC data for Northern hemisphere
        request.set_collection("SW_OPER_MAG"+sats[sc]+"_LR_1B")    
        request.set_products(measurements=["B_NEC"], auxiliaries=['QDLat','QDLon','MLT'],
                             models=[Bmodel], sampling_step="PT1S")
        request.set_range_filter('QDLat', 45, 90)
        data = request.get_between(start_time = large_beg, 
                                   end_time = large_end, asynchronous=True)  
        print('Used MAG L1B file: ', data.sources[1])
        datN_Bnec = data.as_dataframe()
        request.clear_range_filter()   
        # get L2 FAC data for Southern hemisphere
        request.set_range_filter('QDLat', -90, -45)
        data = request.get_between(start_time = large_beg, 
                                   end_time = large_end, asynchronous=True)   
        print('Used MAG L1B file: ', data.sources[1])
        datS_Bnec= data.as_dataframe()    
        request.clear_range_filter()        
        # put toghether data from both hemispheres
        dat = pd.concat([datN_Bnec, datS_Bnec]).sort_index()  

        # append data from different satellites
        Bnec_df.append(dat)            

    # splits the data in quarter orbit sections
    qorbs_Bnec, nrq = ([] for i in range(2))
    for sc in range(nsc):
        # nr. of 1/2 orbits and 1/2 orbit duration 
        nrho = int((orbs[sc,1] - orbs[sc,0] + 1)*2)    
        dtho = (tlarges[sc][1] - tlarges[sc][0])/nrho  
        # start and stop of 1/2 orbits
        begend_hor = [tlarges[sc][0] + ii*dtho for ii in range(nrho +1)]
        # split DataFrame in 1/2 orbit sections; get time of maximum QDLat for each
        horbs = split_into_sections(Bnec_df[sc], begend_hor)
        times_maxQDLat = [horbs[ii]['QDLat'].abs().idxmax().to_pydatetime() \
                        for ii in range(nrho)]
        begend_qor = sorted(times_maxQDLat + begend_hor)
        # split DataFrame in 1/4 orbit sections;
        datq = split_into_sections(Bnec_df[sc], begend_qor)
        qorbs_Bnec.append(datq)
        nrq.append(len(datq))

    # for each sat / quarter orbit computes FAC density
    qorbs_dB, qorbs_fac, badpts = ([[],[],[]] for i in range(3))
    for sc in range(nsc):
        for jj in range(nrq[sc]):
            # store position, magnetic field and magnetic model vectors as data array
            dat_df = qorbs_Bnec[sc][jj]
            ndt = (dat_df.index[-1] - dat_df.index[0]).total_seconds() + 1
            nr_miss_data = ndt - len(dat_df)
            if nr_miss_data:
                 print('NR. OF MISSING DATA FOR Sw'+sats[sc] + ' orbit '+\
                       str(orbs[sc,0]+ jj/4) +': '+ str(nr_miss_data))    
            ind_bads= np.where(np.linalg.norm(np.stack(dat_df['B_NEC'].values), axis=1)==0)[0]
            if len(ind_bads):
                print('NR. OF BAD DATA POINTS FOR Sw'+sats[sc]+ ' orbit '+\
                       str(orbs[sc,0]+ jj/4) +': ', len(ind_bads))
                print(dat_df.index[ind_bads].values)
                dat_df = dat_df.drop(dat_df.index[ind_bads])
            if nr_miss_data or len(ind_bads):
                print('DATA FILTERING MIGHT NOT WORK PROPERLY\n')
            badpts[sc].append(nr_miss_data + len(ind_bads))
            
            ti = dat_df.index
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
            # compute the FAC current and stores data in DataFrames
            j_df = singleJfac(ti, R, B, dB, use_filter = True)              
            j_df['FAC_flt_sup'] = np.where(np.abs(j_df['FAC_flt']) >= jTHR, j_df['FAC_flt'], 0)
            j_df['QDLat'] = np.interp(j_df.index.asi8, ti.asi8, dat_df['QDLat'].values)
            j_df['QDLon'] = np.interp(j_df.index.asi8, ti.asi8, dat_df['QDLon'].values)
            
            colBdB = pd.MultiIndex.from_product([['Rgeo','Bgeo','dBgeo','dBnec'],\
                                    ['x','y','z']], names=['Var','Com'])
            qorbs_dB[sc].append(pd.DataFrame(np.concatenate((R.reshape(-1,3), B.reshape(-1,3), \
                        dB.reshape(-1,3), dBnec.reshape(-1,3)),axis=1), columns=colBdB,index=ti))    
            qorbs_fac[sc].append(j_df)

    # for each sat / quarter orbit computes AO limits and applies MVA
    qimva_df, qorbs_dBmva = qorbsMVA(sats, orbs, qorbs_dB, qorbs_fac)

    # for each quarter orbit computes the optimum time lag and the correlation
    # coefficient between magnetic perturbation on satellites 'A' and 'C'    
    qicc_df, Bcc_df = qorbsCC(sats, qimva_df, qorbs_dBmva)

    param = {'dtime_beg':dtime_beg,'dtime_end':dtime_end,'sats': sats, 'jTHR':jTHR}

    save_qi(qimva_df, qicc_df, param)
    
    if saveplot:
        plot_qi(qorbs_Bnec, qorbs_dB, qorbs_fac, qorbs_dBmva, qimva_df, Bcc_df, qicc_df, param)    

    # prepare a compact output    
    input_df = qorbs_Bnec
    # below one uses swA because is a lower sat. i.e. with smaller orbital period
    RBdBAng_df = [[[] for i in range(nrq[0])] for j in range(nsc)] 
    for sc in range(nsc):
        for jj in range(nrq[sc]):
            tmp_df = qorbs_dB[sc][jj].copy()
            tmp_df.columns = ['_'.join(col) for col in tmp_df.columns.values]
            RBdBAng_df[sc][jj] = tmp_df.join(qorbs_dBmva[sc][jj])
    fac_df = qorbs_fac
    
    return input_df, RBdBAng_df, fac_df, qimva_df, qicc_df, param

def qorbsMVA(sats, orbs, qorbs_dB, qorbs_fac):
    '''
    Automatically estimate the MVA related quality indices of
    FAC structures

    Parameters
    ----------
    sats : list
        satellites entering the analysis, e.g. ['A', 'C'] or ['A', 'B', 'C']
    orbs : array
        array of integers of size [nr. sat, 2], indicate for each satellite
        the start and end orbit numbers
    qorbs_dB : list
        list of DataFrrames (one per satellite per quarter-orbit
        section) with magnetic field perturbation in GEO frame
    fac_df : list
        list of DataFrrames (one per satellite per quarter-orbit
        section) with estimated single=satellite FAC data

    Returns
    -------
    qimva_df : DataFrame
        MVA results, FAC planarity, and FAC inclination
    qorbs_dBmva : list
        list of DataFrrames (one per satellite per quarter-orbit
        section) with magnetic field perturbation in MVA frame
    '''

     # for each sat / quarter orbit computes AO location/extenssion and applies MVA
    nsc = len(sats)
    col_qi = ['sat', 'orbit','TbegMVA', 'TendMVA', 'lmin', 'lmax', 'lrat', 
              'angVN', 'nx', 'ny', 'nz','TcenAO']
    col_Bmva = ['dB_B', 'dB_min', 'dB_max', 'ang_VN']
    qimva_df, qorbs_dBmva = ([] for i in range(2))
    for sc in range(nsc):
        qisc_df = pd.DataFrame(columns = col_qi)
        Bmvasc_df = []
        for jj in range(len(qorbs_fac[sc])):
            tbeg, tend, tcen  = list(find_ao_margins(qorbs_fac[sc][jj])[0:3])
            if pd.isna(tbeg):
                tbeg_cor, tend_cor = tbeg, tend
                ang_vn_jj = np.nan
                eigval, B_unit, maxdir, mindir = (np.full(3,np.nan) for i in range(4))
                Bmvasc_jj = pd.DataFrame(columns = col_Bmva)
            else:
                tbeg_cor = tbeg.ceil(freq = 's')
                tend_cor = tend.floor(freq = 's')                  
                ti_un = qorbs_dB[sc][jj].index.values
                dB_un = qorbs_dB[sc][jj]['dBgeo'].values
                B_un = qorbs_dB[sc][jj]['Bgeo'].values
                R_un = qorbs_dB[sc][jj]['Rgeo'].values
                # eV2d is the satellite velocity in the tangent plane 
                # (i.e. perpendicular to position vector)
                V3d = R_un[1:,:] - R_un[:-1,:] 
                Rmid = 0.5*(R_un[1:,:] + R_un[:-1,:])
                eV3d, eRmid = normvec(V3d), normvec(Rmid)
                eV2d = normvec(np.cross(eRmid, np.cross(eV3d, eRmid)))
                # select quantities for MVA interval and remove NaN points
                indok = np.where((ti_un >= tbeg_cor) & (ti_un <= tend_cor) & \
                                 ~np.isnan(dB_un[:,0]))[0]
                dB_int, B_int = dB_un[indok,:], B_un[indok,:]
                B_ave = np.average(B_int, axis = 0)
                B_unit = B_ave / np.linalg.norm(B_ave)
                # apply constrained MVA
                eigval,eigvec = mva(dB_int, cdir=B_unit)
                # select the minvar orientation according to sat. velocity
                eV3d_ave = np.average(eV3d[indok[:-1]], axis = 0)  
                mindir = eigvec[:,1] 
                if np.sum(mindir*eV3d_ave) < 0:
                    mindir = -eigvec[:,1]
                maxdir = np.cross(B_unit, mindir)
                # compute the FAC inclination wrt sat. velocity in the tangential plane
                eN2d, ang = eV2d.copy(), np.zeros(len(ti_un))
                eN2d[indok[:-1]] = \
                    normvec(np.cross(eRmid[indok[:-1]], np.cross(mindir, eRmid[indok[:-1]])))
                cross_v_n = np.cross(eV2d[indok[:-1]], eN2d[indok[:-1]])
                sign_ang = np.sign(np.sum(eRmid[indok[:-1]]*cross_v_n, axis=-1))
                ang[indok[:-1]]  = \
                    np.degrees(np.arcsin(sign_ang*np.linalg.norm(cross_v_n, axis=-1)))
                ang[0:indok[0]] = ang[indok[0]]
                ang[indok[-1]:] = ang[indok[-2]]
                ang_vn_jj = np.round(np.mean(ang[indok[:-1]]), 1)
                # transform magnetic perturbation in MVA frame
                geo2mva = np.stack((B_unit, mindir, maxdir), axis=1)
                dB_mva = np.matmul(qorbs_dB[sc][jj]['dBgeo'].values, geo2mva)
                Bmvasc_jj = pd.DataFrame(np.concatenate((dB_mva, ang[:,np.newaxis]),axis = 1),\
                                    columns=col_Bmva, index=ti_un)
            qisc_df.loc[len(qisc_df)] = [sats[sc], orbs[sc, 0]+jj/4., tbeg_cor, tend_cor, 
                        round(eigval[1],1), round(eigval[2],1), round(eigval[2]/eigval[1],1), 
                        round(ang_vn_jj, 1), np.round(mindir[0], 3), np.round(mindir[1], 3),
                        np.round(mindir[2], 3), tcen.round(freq='S')]
            Bmvasc_df.append(Bmvasc_jj)
            print('sw'+sats[sc]+' orbit '+str(orbs[sc, 0]+jj/4.)+
                  ' MVA interval: ', tbeg_cor, ' ', tend_cor)
            print('B_unit= %10.2f'%eigval[0], np.round(B_unit, decimals=4))
            print('mindir= %10.2f'%eigval[1], np.round(mindir, decimals=4))
            print('maxdir= %10.2f'%eigval[2], np.round(maxdir, decimals=4))  
            print('')
        qimva_df.append(qisc_df)
        qorbs_dBmva.append(Bmvasc_df) 
    return qimva_df, qorbs_dBmva

def qorbsCC(sats, qimva_df, qorbs_dBmva):
    '''
    Perform correlation analysis between the magnetic field
    perturbations at the lower Swarm satellite

    Parameters
    ----------
    sats : list
        satellites entering the analysis, e.g. ['A', 'C'] or ['A', 'B', 'C']
    qimva_df : DataFrame
        MVA results, FAC planarity, and FAC inclination
    qorbs_dBmva : list
        list of DataFrrames (one per satellite per quarter-orbit section)
        with magnetic field perturbation in MVA frame

    Returns
    -------
    qicc_df : DataFrame
        results from the correlation analysis between the magnetic
        field perturbations at the lower Swarm satellites
    Bcc_df : list
        list of DataFrrames (one per quarter-orbit section)
        with magnetic field perturbation of the reference satellite
        along the maximum magnetic variance direction
    '''

    # for each quarter orbit computes the optimum time lag and the correlation
    # coefficient between magnetic perturbation on satellites 'A' and 'C'
    dt_mva = [[] for i in range(len(sats))]
    for ii in range(len(sats)):
        dt_mva[ii] = qimva_df[ii]['TendMVA'] - qimva_df[ii]['TbegMVA'] 
        
    col_cc = ['refsc', 'Trefbeg', 'Trefend','cc', 'opt_lag', 'orbit_swA', 'orbit_swC']    
    qicc_df = pd.DataFrame(columns = col_cc)
    
    Bcc_df = []
    for jj in range(len(qimva_df[sats.index('A')])):
        iref, isec = sats.index('A'), sats.index('C')
        if (pd.notna(dt_mva[sats.index('A')][jj]) & pd.notna(dt_mva[sats.index('C')][jj])):
            # set the reference s/c (i.e. the one with smaller MVA interval)
            if dt_mva[sats.index('A')][jj] > dt_mva[sats.index('C')][jj]:                
                iref, isec = sats.index('C'), sats.index('A')
            refsc = sats[iref]
            # quarter orbit timeline and data for the second and reference s/c
            tsec, tref = qorbs_dBmva[isec][jj].index, qorbs_dBmva[iref][jj].index
            dBsec = qorbs_dBmva[isec][jj]['dB_max'].values 
            dBref = qorbs_dBmva[iref][jj]['dB_max'].values
            # index range and data of MVA interval for reference s/c
            imva = np.where((tref >= qimva_df[iref]['TbegMVA'].iloc[jj]) & 
                            (tref <= qimva_df[iref]['TendMVA'].iloc[jj]))             
            dBref_mva = qorbs_dBmva[iref][jj]['dB_max'].values[imva]            
            # chooses a common timestamp for both s/c (e.g. the center of tref)
            tstamp = tref.mean().round(freq='S')
            idxstamp_sec = np.where((tsec == tstamp))[0][0]
            # lag of the (start of) reference interval wrt timestamp
            lag_ref = int((tstamp - qimva_df[iref]['TbegMVA'].iloc[jj]).total_seconds())            
            # nr. of points in MVA int
            nmva = int(dt_mva[iref][jj].total_seconds() + 1)
            # choose time-lags between -/+ 30 sec but takes care whether this is possible
            imin = int(np.max([-30, (np.min(tsec) - 
                                     qimva_df[iref]['TbegMVA'].iloc[jj]).total_seconds()]))
            imax = int(np.min([30, (np.max(tsec) - 
                                    qimva_df[iref]['TendMVA'].iloc[jj]).total_seconds()]))             
            nlags = int(imax - imin +1)  # nr. of time lags
            ls_run = np.full((nlags), np.nan) 
            for ii in range(imin, imax+1):
                dBsec_run = dBsec[idxstamp_sec- lag_ref +ii: idxstamp_sec- lag_ref +ii+nmva]
                ls_run[ii-imin] = np.linalg.norm(dBsec_run - dBref_mva)/len(dBref_mva)
            opt_lag_ls = int(np.argmin(ls_run) + imin)
            indsec = idxstamp_sec - lag_ref + opt_lag_ls
            dBsec_opt = dBsec[indsec:indsec + nmva] 
            cc_ls = np.corrcoef(dBsec_opt, dBref_mva)[0,1] 
            Bcc_df.append(pd.DataFrame(dBref_mva,index=tsec[indsec:indsec + nmva]))
        else:
            refsc = 'None'
            opt_lag_ls, cc_ls = np.nan, np.nan
            Bcc_df.append(pd.DataFrame())
        qicc_df.loc[len(qicc_df)] = [refsc, qimva_df[iref]['TbegMVA'].iloc[jj], 
                  qimva_df[iref]['TendMVA'].iloc[jj], np.round(cc_ls, 4), opt_lag_ls, 
                  qimva_df[sats.index('A')]['orbit'].iloc[jj], 
                  qimva_df[sats.index('C')]['orbit'].iloc[jj]]
    
    return qicc_df, Bcc_df
