#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.dates as mdt
from viresclient import set_token
from viresclient import SwarmRequest
import numpy as np
import pandas as pd
from SwarmFACE.utils import *
import warnings
warnings.filterwarnings('ignore')

def save_three_sat_conj(conj_df, param):
    '''Save the Swarm conjunctions to ASCII file. Input from 
    find_3sat_conj.py
    '''
    
    dtime_beg = param['dtime_beg']
    dtime_end = param['dtime_end']    
    delT = param['delT']
    delN = param['delN']
    delE = param['delE']
    
    str_trange = dtime_beg.replace('-','')[:8]+'_'+ dtime_beg.replace(':','')[11:15] \
        + '_'+ dtime_end.replace('-','')[:8] + '_'+ dtime_end.replace(':','')[11:15]
    fname_out = 'swABC_conjunction_'+ str_trange + '.dat'

    a_df = conj_df.copy(deep=True)
    int_cols = ['delTswAB','delTswCB','delNswAB', 'delNswCB','delEswAB','delEswCB']
    for cols in int_cols:
        a_df[cols] = a_df[cols].map('{:7d}'.format)
    flt_cols = ['QDLatA', 'QDLatB', 'QDLatC','QDLonA', 'QDLonB', 'QDLonC']
    for cols in flt_cols:
        a_df[cols] = a_df[cols].map('{:9.2f}'.format)                              
    orbit_cols = ['orbitA', 'orbitB', 'orbitC']
    for cols in orbit_cols:
        a_df[cols] = a_df[cols].map('{:12.2f}'.format)     

    with open(fname_out, 'w') as file:
        file.write('# Time range for searching conjunctions: ' + dtime_beg +\
                   ' - '+ dtime_end + '\n')
        file.write('# Accepted time window (s): ' + str(delT) +'\n')
        file.write('# Accepted separation along North (km): ' + str(delN) + '\n')
        file.write('# Accepted separation along East (km): ' + str(delE) + '\n')           
        file.write('# index, '+ (',  '.join(a_df.columns)) + '\n')          
    a_df.to_csv(fname_out, mode='a', sep=",", date_format='%Y-%m-%d %H:%M:%S', header=False) 
    
def plot_three_sat_conj(conj_df, fac_df, qpar_df, param):
    '''
    Generate the standard plot to show the Swarm conjunction. Input from 
    find_3sat_conj.py
    '''
    
    dtime_beg = param['dtime_beg']
    dtime_end = param['dtime_end']    
    delT = param['delT']
    delN = param['delN']
    delE = param['delE']
    sats = ['A','B','C']
    nsc = len(sats)      # nr. satellites    
    Bmodel = "CHAOS-all='CHAOS-Core'+'CHAOS-Static'+'CHAOS-MMA-Primary'+'CHAOS-MMA-Secondary'"
    marg = pd.to_timedelta(300, unit='s')

    timesB = conj_df['TimeB']
    coldBgeo = pd.MultiIndex.from_product([['dBgeo'],sats,['X','Y','Z']], 
                               names=['Var','Sat','Com'])

    for kk in range(len(conj_df)):     
        df_int = [fac_df[sc][timesB[kk] - marg: timesB[kk] + marg] for sc in range(nsc)]
      
        dBgeo = get_db_data(sats, timesB[kk] - marg, timesB[kk] + marg, Bmodel)
        tbeg = dBgeo.index[0]
        tend = dBgeo.index[-1]
        
        fname_fig = 'swABC_conjunction_'+ timesB[kk].strftime('%Y%m%d_%H%M') +'.eps'

        fig_size = (8.27,11.69)
        fig = plt.figure(figsize=fig_size, frameon=True)
        fig.suptitle('Swarm conjunction interval: ' + tbeg.strftime('%Y-%m-%d') + \
        '   ' + tbeg.strftime('%H:%M') + ' - ' + tend.strftime('%H:%M'), \
             size='xx-large', fontweight = 'demibold', y=0.992)
        
        # PANEL PART
        # designes the panel frames 
        xle, xri, ybo, yto = 0.12, 0.94, 0.03, 0.06     # plot margins
        ratp = np.array([1.2, 0.6, 1.2, 0.6, 1.2, 0.6, 1., 1., 1.])         #relative hight of each panel 
        hsep = np.array([0.015, 0., 0.07, 0., 0.07, 0., 0.08, 0.0, 0.0])        # vspace between panels 
        nrp = len(ratp) 
        
        hun = (1 - yto - ybo - hsep.sum())/ratp.sum() 
        ylo = np.zeros(nrp)     # y low for each panel
        yhi = np.zeros(nrp)     # y high for each panel
        for ii in range(0, nrp): 
            ylo[ii] = (1 - yto) -  ratp[: ii+1].sum()*hun - hsep[: ii+1].sum()
            yhi[ii] = ylo[ii] + hun*ratp[ii]

        # creates axex for each panel
        ax = [0] * nrp
        for ii in range(nrp):
            ax[ii] =  fig.add_axes([xle, ylo[ii], xri-xle, hun*ratp[ii]])
            
        for ii in range(6):    
            ax[ii].set_xlim(timesB[kk] - marg, timesB[kk] + marg)
            ax[ii].set_xticklabels([])  
        
        for ii in range(6,9): 
            ax[ii].get_shared_x_axes().join(ax[6], ax[7], ax[8])

        #gets the QDLat trend, sign, maximum value, and value at central FAC  
        qd_trends, qd_signs, qd_maxs, qd_aocs = (np.zeros(nsc) for i in range(4))
        for sc in range(3):
            qd_trends[sc] = qpar_df[sc].loc[qpar_df[sc]['orbit'] == 
                           conj_df['orbit'+sats[sc]][kk], 'qdlat_trend'].iloc[0]
            qd_signs[sc] = qpar_df[sc].loc[qpar_df[sc]['orbit'] == 
                           conj_df['orbit'+sats[sc]][kk], 'qdlat_sign'].iloc[0] 
            qd_aocs[sc] = qpar_df[sc].loc[qpar_df[sc]['orbit'] == 
                           conj_df['orbit'+sats[sc]][kk], 'aoc_qdlat'].iloc[0]            
            qd_maxs[sc] = df_int[sc]['QDLat'].abs().max()
                       
        # defines the common QDLat range for the last three panels as:
        # - lower limit is the mean QDLat values of the central FACs, minus 7 degree
        # - upper value as +/- maximum QDLat absolute values
        # - the trend and sign is taken from SwarmA
        ss = qd_signs[0]
        tt = qd_trends[0]
        qd_range = ss*np.array([abs(qd_aocs.mean().round()) - 7, np.ceil(qd_maxs.max())])
        if ss*tt < 0:
            qd_range = np.flip(qd_range)

        # Plots title    
        ax[0].set_title('\nswA - swB   delT [s]: '+ str(conj_df['delTswAB'][kk]) +\
                        ',   delN [km]: ' + str(conj_df['delNswAB'][kk])+ \
                        ',   delE [km]: ' + str(conj_df['delEswAB'][kk])+ \
                        '\nswC - swB   delT [s]: '+ str(conj_df['delTswCB'][kk]) +\
                        ',   delN [km]: ' + str(conj_df['delNswCB'][kk])+ \
                        ',   delE [km]: ' + str(conj_df['delEswCB'][kk]), 
                        fontsize = 'x-large', pad = 8) 
        
        for sc in range(3):
            df_int_sc = df_int[sc]
            # plot the fpanels with dBgeo data
            ax[sc*2].plot(dBgeo[('dBgeo',sats[sc])])
            ax[sc*2].set_ylabel('$dB_{GEOC}$ sw'+sats[sc]+'\n[nT]', linespacing=1.7)
            ax[sc*2].axvline(conj_df['Time'+sats[sc]][kk], ls='--', c='k')
            ax[sc*2].axvline(conj_df['TimeB'][kk], ls='--', c='r')              
            ax[sc*2].axvline(df_int_sc['QDLat'].abs().idxmax(), ls='-', c='b')

            # plots the fpanels with filtered FAC data    
            ax[sc*2 + 1].plot(df_int_sc['FAC_flt_sup'], linewidth=2)
            ax[sc*2 + 1].set_ylabel('$J_{FAC}$\n[$\mu A/m^2$]', linespacing=1.7)
            ax[sc*2 + 1].axvline(conj_df['Time'+sats[sc]][kk], ls='--', c='k')
            ax[sc*2 + 1].axvline(conj_df['TimeB'][kk], ls='--', c='r')             
            ax[sc*2 + 1].axvline(df_int_sc['QDLat'].abs().idxmax(), ls='-', c='b')
            ax[sc*2 + 1].axhline(0, ls='--', c='k')

            # adds QDLat, QDLon and MLT tick labels    
            locx = ax[sc*2 + 1].get_xticks()
            qdlat_ipl = np.round(np.interp(locx, mdt.date2num(df_int_sc.index), \
                                    df_int_sc['QDLat']), decimals=2).astype('str')
            qdlon_ipl = np.round(np.interp(locx, mdt.date2num(df_int_sc.index), \
                                    df_int_sc['QDLon']), decimals=2).astype('str')
            mlt_ipl = np.round(np.interp(locx, mdt.date2num(df_int_sc.index), \
                                    df_int_sc['MLT']), decimals=1).astype('str')
            lab_fin = ['']*len(locx)
            for ix in range(len(locx)):
                lab_ini = mdt.num2date(locx[ix]).strftime('%H:%M:%S')
                lab_fin[ix] = lab_ini + '\n' +qdlat_ipl[ix] + '\n' + \
                            qdlon_ipl[ix] + '\n' + mlt_ipl[ix]
            ax[sc*2 + 1].set_xticklabels(lab_fin)
            plt.figtext(0.01, ylo[sc*2 + 1]-0.008, 'Time\nQDLat\nQDLon\nMLT', va='top')
            plt.figtext(0.96, (ylo[sc*2 + 1]+yhi[sc*2])/2, 'Swarm '+ sats[sc], va='center', \
                    rotation=90, size='x-large', fontweight = 'medium')
            
            # computes dBgeo as function of QDLat. For that takes the start and stop 
            # times for the quarter-orbit and applies find_jabs_midcsum to computes 
            # the evolution of time as a function of QDLat
            tbeg_qor = qpar_df[sc].loc[qpar_df[sc]['orbit'] == 
                           conj_df['orbit'+sats[sc]][kk], 'beg_time'].iloc[0]            
            tend_qor = qpar_df[sc].loc[qpar_df[sc]['orbit'] == 
                           conj_df['orbit'+sats[sc]][kk], 'end_time'].iloc[0]  

            qorb_fac = fac_df[sc][tbeg_qor:tend_qor]    # quarter-orbit data
            ti_arr, qd_arr = find_ao_margins(qorb_fac)[-2:] 
            
            cmp = ['X', 'Y', 'Z']
            db_arr = np.zeros((len(ti_arr),3))
            for ic in range(3):
                db_arr[:,ic] = np.interp(ti_arr, dBgeo[('dBgeo',sats[sc])].index.values.astype(float), \
                           dBgeo[('dBgeo',sats[sc],cmp[ic])].values)
            aoc_time = qpar_df[sc].loc[qpar_df[sc]['orbit'] == 
                             conj_df['orbit'+sats[sc]][kk], 'aoc_time'].iloc[0]
            ind_qd = (np.abs(ti_arr - aoc_time.asm8.astype("float"))).argmin()            
            
            # plots the last three panels with dBgeo as function of QDLat.
            ax[sc + 6].plot(qd_arr, db_arr)
            ax[sc + 6].set_ylabel('$dB_{GEOC}$ sw'+sats[sc]+'\n[nT]', linespacing=1.7)
            ax[sc + 6].axvline(qd_aocs[sc], ls='--', c='k')
            ax[sc + 6].axhline(0, ls='--', c='k')
            ax[sc + 6].xaxis.set_major_locator(plt.MaxNLocator(10))
            ax[sc + 6].set_xlim(qd_range)
            if sc in range(2):
                ax[sc + 6].set_xticklabels([])
                ax[sc + 6].get_shared_y_axes().join(ax[6], ax[7], ax[8])       
            plt.figtext(0.01, ylo[8]-0.008, 'QDLat', va='top')

        plt.draw()
        fig.savefig(fname_fig)
   