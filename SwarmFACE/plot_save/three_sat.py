#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.dates as mdt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import SwarmFACE.esaL2 as esaL2

def save_three_sat(j_df, param):
    '''
    Save to ASCII file results from three-satellite algorithm. Input from 
    j3sat.py
    '''            

    dtime_beg = param['dtime_beg']
    dtime_end = param['dtime_end']    
    sats = param['sats']
    angTHR = param['angTHR']
    tshift = param['tshift']
    use_filter = param['use_filter']
    Bmodel = param['Bmodel']    
    timebads = param['timebads']
    angBN = j_df['angBN'].values
    
    nbads = np.zeros(3, dtype=int)
    for ii in range(3):
        nbads[ii] = 0 if timebads['sc'+str(ii)] is None else len(timebads['sc'+str(ii)])
 
    str_trange = dtime_beg.replace('-','')[:8]+'_'+ \
         dtime_beg.replace(':','')[11:17] + '_'+dtime_end.replace(':','')[11:17]
    
    fname_out = 'FAC_LR_sw'+''.join(sats)+'_'+str_trange +'.dat'
       
    # exports the results
    out_df = j_df[['Rmid_x','Rmid_y','Rmid_z','FAC','Jn','FAC_er', 'Jn_er',
                  'angBN','CN3']]  
    text_beg = '# time [YYYY-mm-ddTHH:MM:SS.f],\t  Rmid_x [km],\t  Rmid_y,\t\
      Rmid_z,\t  Jfac [microA/m^2],\t  Jn,\t  errJfac,\t  errJn,\t  '
        
    if use_filter:
        out_df = j_df
        text_beg = text_beg + 'Jfac_flt,\t  Jn_flt,\t  errJfac_flt,\t  errJn_flt,\t  '         
    text_header = text_beg + 'angBN [deg],\t  CN3 \n'
    
    with open(fname_out, 'w') as file:
        file.write('# Swarm sat: [' + ', '.join(sats) + ' ] \n')
        file.write('# Time-shift in sec.: [' + ', '.join(str(x) for x in tshift) + '] \n')  
        file.write('# Accepted B - s/c plane angle threshold: ' + str(angTHR) + ' deg \n')                     
        file.write('# Model: ' + Bmodel + '\n')
        file.write('# use filter: ' + str(use_filter) + '\n')                   
        file.write('# Number of bad data points in L1b files: ' + str(nbads.sum()) + '\n')
        for ii in range(3):
            if nbads[ii]:
                file.write('#   Sw'+sats[ii] + ' ' + str(nbads[ii]) + ' point(s) : '+ 
                    '; '.join(timebads['sc'+str(ii)].strftime('%H:%M:%S').values) + '\n')
        file.write(text_header)
    out_df.to_csv(fname_out, mode='a', sep=",", na_rep = 'NaN', date_format='%Y-%m-%dT%H:%M:%S.%f',\
                  float_format='%15.4f', header=False)
        
def plot_three_sat(j_df, dat_df, param):
    '''
    Plot results from three-satellite algorithm. Input from 
    j3sat.py
    '''        

    dtime_beg = param['dtime_beg']
    dtime_end = param['dtime_end']    
    sats = param['sats']
    angBN = j_df['angBN'].values
    angTHR = param['angTHR']
    tshift = param['tshift']
    use_filter = param['use_filter']
    Bmodel = param['Bmodel']    
    timebads = param['timebads']
    
    nsc = len(sats)
    nbads = np.zeros(3, dtype=int)
    for ii in range(3):
        nbads[ii] = 0 if timebads['sc'+str(ii)] is None else len(timebads['sc'+str(ii)])    
    
    FAC_L2 = esaL2.dual(dtime_beg, dtime_end)
    
    str_trange = dtime_beg.replace('-','')[:8]+'_'+ \
         dtime_beg.replace(':','')[11:17] + '_'+dtime_end.replace(':','')[11:17]   
         
    fname_fig = 'FAC_LR_sw'+''.join(sats)+'_'+str_trange +'.eps'    
    
    bad_ang = np.less(np.abs(np.cos(angBN*np.pi/180.)), np.cos(np.deg2rad(90 - angTHR)))
    
    line1_text = 'Time interval:  ' + dtime_beg[:10]+'  ' + \
                dtime_beg[11:19] + ' - '+dtime_end[11:19] 
    line2_text = 'sat: [' + ', '.join(sats) + ']    Time-shift: [' + \
            ', '.join(str(x) for x in tshift) + ']     bad points: '+str(nbads.sum()) \
            + '     small angle points: ' +str(len(np.where(bad_ang)[0]))
    
    # creates fig and axes objects
    fig_size = (8.27,11.69)
    fig = plt.figure(figsize=fig_size, frameon=True)
    fig.suptitle('FAC density estimate with three-satellite method\n', size='xx-large',\
                 weight = 'bold', y=0.99)
    plt.figtext(0.5, 0.962, line1_text, fontsize = 'x-large', \
                va='top', ha = 'center')
    plt.figtext(0.5, 0.94, line2_text, fontsize = 'x-large', \
                va='top', ha = 'center')

    # PANEL PART
    # designes the panel frames 
    xle, xri, ybo, yto = 0.125, 0.94, 0.3, 0.09     # plot margins
    ratp = np.array([1, 1, 1, 0.7, 0.7, 1.5, 0.7 ]) #relative hight of each panel 
    hsep = 0.005                                    # vspace between panels 
    nrp = len(ratp) 
    
    hun = (1-yto - ybo - (nrp-1)*hsep)/ratp.sum() 
    yle = np.zeros(nrp)     # y left for each panel
    yri = np.zeros(nrp) 
    for ii in range(nrp): 
        yle[ii] = (1 - yto) -  ratp[: ii+1].sum()*hun - ii*hsep
        yri[ii] = yle[ii] + hun*ratp[ii]
    
    # creates axex for each panel
    ax = [0] * nrp
    for ii in range(nrp):
        ax[ii] =  fig.add_axes([xle, yle[ii], xri-xle, hun*ratp[ii]])
        ax[ii].set_xlim(pd.Timestamp(dtime_beg), pd.Timestamp(dtime_end))
      
    for ii in range(nrp -1):
        ax[ii].set_xticklabels([])
        
    #Plots time-series quantities    
    ax[0].plot(dat_df[('dBgeo','A')])    
    ax[0].set_ylabel('$dB_{GEO}$ swA' +'\n[nT]', linespacing=1.7)
    ax[0].legend(['dB_x', 'dB_y', 'dB_z' ], loc=(0.95, 0.1), handlelength=1)    
    
    ax[1].plot(dat_df[('dBgeo','B')])
    ax[1].get_shared_y_axes().join(ax[0], ax[1], ax[2])
    ax[1].set_ylabel('$dB_{GEO}$ swB' +'\n[nT]', linespacing=1.7)
    ax[1].legend(['dB_x', 'dB_y', 'dB_z' ], loc=(0.95, 0.1), handlelength=1)   

    ax[2].plot(dat_df[('dBgeo','C')])
    ax[2].get_shared_y_axes().join(ax[0], ax[1], ax[2])
    ax[2].set_ylabel('$dB_{GEO}$ swC' +'\n[nT]', linespacing=1.7)
    ax[2].legend(['dB_x', 'dB_y', 'dB_z' ], loc=(0.95, 0.1), handlelength=1)
    
    ax[3].plot(j_df['CN3'])
    ax[3].set_ylabel('log(CN3)\n')
    
    ax[4].plot(j_df['angBN'])
    ax[4].set_ylabel('angBN\n[deg]\n', linespacing=1.7)
    
    ax[5].plot(j_df['FAC'], label='$\mathrm{J_{ABC}}$')
    if use_filter:
        ax[5].plot(j_df['FAC_flt'], label='$\mathrm{J_{ABC \_ flt}}$')
    ax[5].plot(FAC_L2['FAC'], label='$\mathrm{J_{Level2}}$')
    ax[5].axhline(y=0, linestyle='--', color='k', linewidth=0.7)
    ax[5].set_ylabel('$J_{FAC}$\n[$\mu A/m^2$]', linespacing=1.7)
    ax[5].legend(loc = (0.95, 0.1), handlelength=1)
   
    ax[6].plot(j_df['FAC_er'], label='$\mathrm{J_{ABC \_ er}}$')
    if use_filter:
        ax[6].plot(j_df['FAC_flt_er'], label='$\mathrm{J_{ABC \_ flt \_ er}}$')    
    ax[6].set_ylabel(r'$J_{FAC\_er}}$'+'\n'+'[$\mu A/m^2$]', linespacing=1.7)
    ax[6].set_ylim(bottom = -0.05, top = 1.)
    ax[6].legend(loc = (0.95, 0.1), handlelength=1)
    
    for ii in range(1,nrp -1):
        ax[ii].get_shared_x_axes().join(ax[0])
    
    # creates the ephemerides    
    latc = dat_df[('Rsph','C','Lat')].values
    lonc = dat_df[('Rsph','C','Lon')].values
    locx = ax[nrp-1].get_xticks()
    latc_ipl = np.round(np.interp(locx, mdt.date2num(dat_df.index), \
                                latc), decimals=2).astype('str')
    lonc_ipl = np.round(np.interp(locx, mdt.date2num(dat_df.index), \
                                lonc), decimals=2).astype('str')
    qdlat_ipl = np.round(np.interp(locx, mdt.date2num(dat_df.index), \
                                dat_df[('QDref','C','QDLat')]), decimals=2).astype('str')
    qdlon_ipl = np.round(np.interp(locx, mdt.date2num(dat_df.index), \
                                dat_df[('QDref','C','QDLon')]), decimals=2).astype('str')
    mlt_ipl = np.round(np.interp(locx, mdt.date2num(dat_df.index), \
                                dat_df[('QDref','C','MLT')]), decimals=1).astype('str')
    
    lab_fin = ['']*len(locx)
    for ii in range(len(locx)):
        lab_ini = mdt.num2date(locx[ii]).strftime('%H:%M:%S')
        lab_fin[ii] = lab_ini + '\n' +latc_ipl[ii] + '\n' + lonc_ipl[ii] + \
        '\n'+ qdlat_ipl[ii] +'\n' +qdlon_ipl[ii] + '\n' + mlt_ipl[ii]
        
    ax[nrp-1].set_xticklabels(lab_fin)
    plt.figtext(0.01, 0.213, 'Time\nLat\nLon\nQLat\nQLon\nMLT')
 
    # INSET PART
    # designes the insets to plot s/c constellation 
    xmar, ymar, xsep = 0.13, 0.04, 0.05
    xwi = (1 - 2*xmar - 2*xsep)/3.      # inset width
    rat = fig_size[0]/fig_size[1]       # useful to fix the same scales on x and y 
    
    # creates axes for each panel
    ax_conf = [0, 0, 0.]
    for ii in range(3):
        ax_conf[ii] =  fig.add_axes([xmar + ii*(xwi+xsep), ymar, xwi, xwi*rat])
    
    Rc = j_df[['Rmid_x', 'Rmid_y', 'Rmid_z']].values
    R = dat_df['Rgeo'].values.reshape(-1,3,3)
    Rmeso = R - Rc[:, np.newaxis, :]
    
    ic=np.array([1, len(j_df)//2, len(j_df)-2])   # indexes for ploting the s/c
             
    Ri_nec = np.full((len(ic),nsc,3),np.nan)
    Vi_nec = np.full((len(ic),nsc,3),np.nan)
    nlim_nec = np.zeros((len(ic), 2))
    elim_nec = np.zeros((len(ic), 2))
    
    z_geo = np.array([0., 0., 1.])
    for ii in range(len(ic)):
        pos_i = ic[ii]
        # computes NEC associated with the mesocenter location Rc 
        Ci_unit = -Rc[pos_i,:]/np.linalg.norm(Rc[pos_i,:])[...,None]
        Ei_geo = np.cross(Ci_unit, z_geo)
        Ei_unit = Ei_geo/np.linalg.norm(Ei_geo)
        Ni_geo = np.cross(Ei_unit, Ci_unit)
        Ni_unit = Ni_geo/np.linalg.norm(Ni_geo)
        trmat_i = np.stack((Ni_unit, Ei_unit, Ci_unit))
        # transform the sats. position from GEO mesocentric to NEC mesocentric
        Ri_geo = Rmeso[pos_i, :, :]
        Ri_nec[ii, :, :] = np.matmul(trmat_i, Ri_geo[...,None]).reshape(Ri_geo.shape)
        nlim_nec[ii,:] = [max(Ri_nec[ii, :, 0]), min(Ri_nec[ii, :, 0])]
        elim_nec[ii,:] = [max(Ri_nec[ii, :, 1]), min(Ri_nec[ii, :, 1])]  
        Vi_geo = (R[pos_i+1, :, :] - R[pos_i-1, :, :])/2.
        Vi_geo_unit = Vi_geo/np.linalg.norm(Vi_geo,axis=-1)
        Vi_nec[ii, :, :] = np.matmul(trmat_i, Vi_geo_unit[...,None]).reshape(Vi_geo.shape)
        
    # computes the (common) range along N and E
    dn_span = max(nlim_nec[:,0] - nlim_nec[:,1])
    de_span = max(elim_nec[:,0] - elim_nec[:,1])
    d_span = max(np.concatenate((nlim_nec[:,0] - nlim_nec[:,1], 
                                 elim_nec[:,0] - elim_nec[:,1])))*1.2
    
    # plots the s/c positions
    icolor = ['b', 'g', 'r']
    for ii in range(len(ic)):
        norig = np.mean(nlim_nec[ii, :])
        eorig = np.mean(elim_nec[ii, :])    
        ax_conf[ii].set_xlim( norig - d_span/2., norig + d_span/2. )
        ax_conf[ii].set_ylim( eorig - d_span/2., eorig + d_span/2. )   
        for jj in range(3):
            ax_conf[ii].scatter(Ri_nec[ii, jj, 0], Ri_nec[ii, jj, 1], \
               c=icolor[jj], label='sw'+str(sats[jj]))
            ax_conf[ii].arrow(Ri_nec[ii, jj, 0] , Ri_nec[ii, jj, 1], \
                   d_span/10*Vi_nec[ii, jj, 0], d_span/10*Vi_nec[ii, jj, 1], \
                   color=icolor[jj], head_width = 4)
    
    ax_conf[0].set_ylabel('East [km]')
    ax_conf[0].set_xlabel('North [km]')
    ax_conf[1].set_xlabel('North [km]')
    ax_conf[1].set_yticklabels([])
    ax_conf[2].set_xlabel('North [km]')
    ax_conf[2].set_yticklabels([])
    ax_conf[2].legend( loc = (0.98, 0.7), handlelength=1, ncol = 1)
        
    plt.draw()
    fig.savefig(fname_fig)    