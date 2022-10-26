#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.dates as mdt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import SwarmFACE.esaL2 as esaL2

def save_single_sat(j_df, param):
    '''
    Save to ASCII file results from single-satellite algorithm. Input from 
    j1sat.py
    '''          

    dtime_beg = param['dtime_beg']
    dtime_end = param['dtime_end']    
    res = param['res']
    sat = param['sat']
    angBR = j_df['angBR'].values
    angTHR = param['angTHR']
    tincl = param['tincl']
    use_filter = param['use_filter']
    Bmodel = param['Bmodel']    
    timebads = param['timebads']
    
    nbads=0 if timebads is None else len(timebads)
    
    str_trange = dtime_beg.replace('-','')[:8]+'_'+ \
         dtime_beg.replace(':','')[11:17] + '_'+dtime_end.replace(':','')[11:17]
    
    fname_out = 'FAC_'+res+'_sw'+sat[0]+'_'+str_trange +'.dat'
       
    # exports the results
    out_df = j_df[['Rmid_x','Rmid_y','Rmid_z','FAC','IRC','FAC_er', 'IRC_er',
                  'angBR','incl']]  
    text_beg = '# time [YYYY-mm-ddTHH:MM:SS.f],\t  Rmid_x [km],\t  Rmid_y,\t\
      Rmid_z,\t  Jfac [microA/m^2],\t  IRC,\t  errJfac,\t  errJirc,\t  '
        
    if use_filter:
        out_df = j_df
        text_beg = text_beg + 'Jfac_flt,\t  Jirc_flt,\t  errJfac_flt,\t  errJirc_flt,\t  '         
    text_header = text_beg + 'angBR [deg],\t  incl \n'
    
    with open(fname_out, 'w') as file:
        file.write('# Swarm sat: ' + str(sat[0]) + '\n')
        file.write('# Model: ' + Bmodel + '\n')
        file.write('# use filter: ' + str(use_filter) + '\n')                   
        if res == 'LR':
            file.write('# Number of bad data points in L1b file: ' + str(nbads) + '\n')
            if nbads:
                file.write('#' + '; '.join(timebads.strftime('%H:%M:%S').values) + '\n')
        file.write(text_header)
    out_df.to_csv(fname_out, mode='a', sep=",", na_rep = 'NaN', date_format='%Y-%m-%dT%H:%M:%S.%f',\
                  float_format='%15.4f', header=False)
    

def plot_single_sat(j_df, dat_df, param):
    '''
    Plot results from single-satellite algorithm. Input from 
    j1sat.py
    '''    

    dtime_beg = param['dtime_beg']
    dtime_end = param['dtime_end']    
    res = param['res']
    sat = param['sat']
    angBR = j_df['angBR'].values
    angTHR = param['angTHR']
    tincl = param['tincl']
    use_filter = param['use_filter']
    Bmodel = param['Bmodel']    
    timebads = param['timebads']
    
    nbads= 0 if timebads is None else len(timebads)
    
    FAC_L2 = esaL2.single(dtime_beg, dtime_end, sat)
    
    str_trange = dtime_beg.replace('-','')[:8]+'_'+ \
         dtime_beg.replace(':','')[11:17] + '_'+dtime_end.replace(':','')[11:17]    
    
    fname_fig = 'FAC_'+res+'_sw'+sat[0]+'_'+str_trange +'.eps'
    
    bad_ang = np.less(np.abs(np.cos(angBR*np.pi/180.)), np.cos(np.deg2rad(90 - angTHR)))
    if res == 'LR':
        info_points = '\nres.: LR     bad points: '+str(nbads) + '      ' +\
                'low-latitude points: ' +str(len(np.where(bad_ang)[0]))
    else:
        info_points = '\nres.: HR     low-lat. points: '  +str(len(np.where(bad_ang)[0]))                
    add_text = 'Time interval:  ' + dtime_beg[:10]+'  ' + \
                dtime_beg[11:19] + ' - '+dtime_end[11:19] + info_points
    if tincl is not None:
        add_text = add_text + '\nInclination interval:  ' + \
            tincl[0].strftime('%H:%M:%S') + ' - ' + \
            tincl[1].strftime('%H:%M:%S')

    # creates fig and axes objects
    fig_size = (8.27,11.69)
    fig = plt.figure(figsize=fig_size, frameon=True)
    fig.suptitle('FAC density estimate with Swarm '+sat[0]+'\n', size='xx-large',\
                 weight = 'bold', y=0.99)        
    plt.figtext(0.21, 0.96, add_text, fontsize = 'x-large', \
                va='top', ha = 'left')
    
    xle, xri, ybo, yto = 0.125, 0.92, 0.095, 0.105     # plot margins
    ratp = np.array([1, 1, 1, 1, 0.5, 0.5, 0.5])          #relative hight of each panel 
    hsep = 0.005                                    # vspace between panels 
    nrp = len(ratp) 
    
    hun = (1-yto - ybo - (nrp-1)*hsep)/ratp.sum() 
    yle = np.zeros(nrp)     # y left for each panel
    yri = np.zeros(nrp) 
    for ii in range(nrp): 
        yle[ii] = (1 - yto) -  ratp[: ii+1].sum()*hun - ii*hsep
        yri[ii] = yle[ii] + hun*ratp[ii]
    
    # creates axes for each panel
    ax = [0] * nrp
    for ii in range(nrp):
        ax[ii] =  fig.add_axes([xle, yle[ii], xri-xle, hun*ratp[ii]])
        ax[ii].set_xlim(pd.Timestamp(dtime_beg), pd.Timestamp(dtime_end))
      
    for ii in range(nrp -1):
        ax[ii].set_xticklabels([])
        
    #Plots time-series quantities    
    ax[0].plot(dat_df[['dB_xgeo','dB_ygeo','dB_zgeo']])
    ax[0].set_ylabel('$dB_{GEO}$ sw'+sat[0]+'\n[nT]', linespacing=1.7)
    ax[0].legend(['dB_x', 'dB_y', 'dB_z' ], loc='upper right')
    
    ax[1].plot(j_df['FAC'], label='FAC')
    if use_filter:
        ax[1].plot(j_df['FAC_flt'], label='FAC_flt')
    ax[1].set_ylabel(r'$J_{FAC}$'+'\n'+r'$[\mu A/m^2]$', linespacing=1.7)
    ax[1].legend(loc='upper right')
    
    ax[2].plot(j_df['IRC'], label='IRC')
    if use_filter:
        ax[2].plot(j_df['IRC_flt'], label='IRC_flt')
    ax[2].set_ylabel(r'$J_{IRC}$'+'\n'+r'$[\mu A/m^2]$', linespacing=1.7)
    ax[2].legend(loc='upper right') 
    
    ax[3].plot(j_df['FAC'], label='FAC')
    ax[3].plot(FAC_L2['FAC'], label='FAC_Level2')
    ax[3].set_ylabel(r'$J_{FAC}$'+'\n'+r'$[\mu A/m^2]$', linespacing=1.7)
    ax[3].legend(loc='upper right') 

    ax[4].plot(j_df['FAC_er'], label='FAC_er')
    ax[4].plot(j_df['FAC_flt_er'], label='FAC_flt_er')
    ax[4].set_ylabel(r'$J_{FAC\_er}$'+'\n'+r'$[\mu A/m^2]$', linespacing=1.7)
    ax[4].legend(loc='upper right')
    
    ax[5].plot(j_df['angBR'])
    ax[5].set_ylabel('ang. BR'+'\n'+r'$[deg]$', linespacing=1.7)
#    ax[5].legend() 
    
    ax[6].plot(j_df['incl'])
    ax[6].set_ylabel('incl.'+'\n'+r'$[deg]$', linespacing=1.7)
#    ax[6].legend() 

    for ii in range(1,nrp -1):
        ax[ii].get_shared_x_axes().join(ax[0])

    if tincl is not None:
        for ii in range(nrp):
            ax[ii].axvline(tincl[0], ls='--', c='k')
            ax[ii].axvline(tincl[1], ls='--', c='k')

    # creates the ephemerides    
    latc = dat_df['Latitude'].values
    lonc = dat_df['Longitude'].values
    locx = ax[nrp-1].get_xticks()
    latc_ipl = np.round(np.interp(locx, mdt.date2num(dat_df.index), \
                                latc), decimals=2).astype('str')
    lonc_ipl = np.round(np.interp(locx, mdt.date2num(dat_df.index), \
                                lonc), decimals=2).astype('str')
    qdlat_ipl = np.round(np.interp(locx, mdt.date2num(dat_df.index), \
                                dat_df['QDLat']), decimals=2).astype('str')
    qdlon_ipl = np.round(np.interp(locx, mdt.date2num(dat_df.index), \
                                dat_df['QDLon']), decimals=2).astype('str')
    mlt_ipl = np.round(np.interp(locx, mdt.date2num(dat_df.index), \
                                dat_df['MLT']), decimals=1).astype('str')
    
    lab_fin = ['']*len(locx)
    for ii in range(len(locx)):
        lab_ini = mdt.num2date(locx[ii]).strftime('%H:%M:%S')
        lab_fin[ii] = lab_ini + '\n' +latc_ipl[ii] + '\n' + lonc_ipl[ii]+ \
        '\n'+ qdlat_ipl[ii] + '\n' +qdlon_ipl[ii] + '\n' + mlt_ipl[ii]
        
    ax[nrp-1].set_xticklabels(lab_fin)
    plt.figtext(0.01, 0.01, 'Time\nLat\nLon\nQLat\nQLon\nMLT')
    
    plt.draw()
    fig.savefig(fname_fig)
    