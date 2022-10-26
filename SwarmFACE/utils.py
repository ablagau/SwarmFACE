#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import signal
from viresclient import SwarmRequest

def normvec(v):
    '''
    Return array of unit vectors

    Parameters
    ----------
    v : numpy.array
        vectors
    Returns
    -------
    numpy.array of unit vectors
    '''

    return np.divide(v,np.linalg.norm(v,axis=-1).reshape(-1,1))

def rotvecax(v, ax, ang):
    '''
    Rotates vector v by angle ang around a normal vector ax.
    Uses Rodrigues' formula when v is normal to ax

    Parameters
    ----------
    v : numpy.array
        vector(s)
    ax : numpy.array
        normal vector(s)
    ang : numpy.array
        angle(s)

    Returns
    -------
    vector or array of vectors
    '''

    sa, ca = np.sin(np.deg2rad(ang)), np.cos(np.deg2rad(ang))
    return v*ca[...,np.newaxis] + np.cross(ax, v)*sa[...,np.newaxis]

def sign_ang(V, N, R):
    '''
    Return the signed angle between vectors V and N, perpendicular
    to R; positive sign corresponds to right hand rule along R

    Parameters
    ---------
    V, N, R : numpy.array
        vectors

    Return
    ------
    angle(s)
    '''

    VxN = np.cross(V, N)
    pm = np.sign(np.sum(R*VxN, axis=-1))
    return np.degrees(np.arcsin(pm*np.linalg.norm(VxN, axis=-1)))  

def sign_dang(ang2, ang1):
    '''
    Return the difference between two angles in [-180, 180]
    deg. interval

    Parameters
    ---------
    ang1, ang2 : numpy.array
        angles in degrees

    Return
    ------
    numpy.array of angles
    '''

    a = ang2-ang1
    return (a + 180) % 360 - 180
  
def R_in_GEOC(Rsph):
    '''
    Returns Swarm position in cartesian GEO frame and
    the rotation matrix from NEC to GEO

    Parameters
    ---------
    Rsph : numpy.array
        satellite position in spherical GEO frame

    Return
    ------
    R : numpy.array
        satellite position in cartesian GEO frame
    : numpy.array
        array of rotation matrices from NEC to GEO
    '''

    latsc = np.deg2rad(Rsph[:,0])
    lonsc = np.deg2rad(Rsph[:,1])  
    radsc = 0.001*Rsph[:,2]
    # prepares conversion to global cartesian frame
    clt,slt = np.cos(latsc.flat),np.sin(latsc.flat)
    cln,sln = np.cos(lonsc.flat),np.sin(lonsc.flat)
    north = np.stack((-slt*cln,-slt*sln,clt),axis=-1)
    east = np.stack((-sln,cln,np.zeros(cln.shape)),axis=-1)
    center = np.stack((-clt*cln,-clt*sln,-slt),axis=-1)
    # stores cartesian position vectors in position data matrix R
    R = -radsc[...,None]*center
    return R, np.stack((north,east,center),axis=-1)

def get_db_data(sats, tbeg, tend, Bmodel, frame = 'GEOC'):
    '''
    Return Swarm magnetic perturbation in a DataFrame structure
    
    Parameters
    ----------
    sats : str or list of str
        satellites
    tbeg, tend : datetime-like
        start and stop times
    Bmodel : string
        magnetic field model used to compute the magnetic
        field perturbations (currently CHAOS)
    frame : string
        reference frame for the output, i.e. 'GEOC' for GEO frame 
        or None for NEC frame
        
    Returns
    -------
    dB : DataFrame
        Swarm magnetic field perturbation data 
    '''

    dti = pd.date_range(start = tbeg.round('s'), 
                        end = tend.round('s'), freq='s', closed='left')
    ndti = len(dti)
    nsc = len(sats)
    Rsph, R, Bnec, Bgeo, Bmod, dBnec, dBgeo = (np.full((ndti,nsc,3),np.nan) for i in range(7))
    request = SwarmRequest()
    for sc in range(nsc):
        request.set_collection("SW_OPER_MAG"+sats[sc]+"_LR_1B")
        request.set_products(measurements=["B_NEC"], 
                             models=[Bmodel],
                             residuals=False, sampling_step="PT1S")
        data = request.get_between(start_time = tbeg, 
                                   end_time = tend,
                                   asynchronous=False)   
        print('Used MAG L1B file: ', data.sources[1])
        dat = data.as_dataframe()
        dsi = dat.reindex(index=dti, method='nearest')
        # store magnetic field perturbation in a data matrices
        Bnec[:,sc,:] = np.stack(dsi['B_NEC'].values, axis=0)
        Bmod[:,sc,:] = np.stack(dat['B_NEC_CHAOS-all'].values, axis=0)  
        dBnec[:,sc,:] = Bnec[:,sc,:] - Bmod[:,sc,:]      
        Rsph[:,sc,:] = dsi[['Latitude','Longitude','Radius']].values
        
    coldBnec = pd.MultiIndex.from_product([['dBnec'],sats,['N','E','C']],
                                          names=['Var','Sat','Com'])        
    dB = pd.DataFrame(dBnec.reshape(-1,nsc*3), columns=coldBnec,index=dti)
    if frame == 'GEOC':
        for sc in range(nsc):
            R[:,sc,:], MATnec2geo_sc = R_in_GEOC(np.squeeze(Rsph[:,sc,:]))  
            Bgeo[:,sc,:] = \
                np.matmul(MATnec2geo_sc,np.squeeze(Bnec[:,sc,:])[...,None]).reshape(-1,3)
            dBgeo[:,sc,:] = \
                np.matmul(MATnec2geo_sc,np.squeeze(dBnec[:,sc,:])[...,None]).reshape(-1,3)
        coldBgeo = pd.MultiIndex.from_product([['dBgeo'],sats,['X','Y','Z']], 
                               names=['Var','Sat','Com'])
        dB = pd.DataFrame(dBgeo.reshape(-1,nsc*3), columns=coldBgeo,index=dti)
    return dB

def GapsAsNaN(df_ini, ind_gaps):
    '''
    Given a DataFrame object, replace with NaN the magnetic field data 
    and model for certain location
    
    Parameters
    ----------
    df_ini : DataFrame
        containes columns with magetic field data and model
    ing_gaps : numpy.array
        integer-location based indexing
        
    Returns
    -------
    df_out : DataFrame
        new object with NaN 
    : index
        index values at replaced location    
    '''
    df_out = df_ini.copy()
    df_out['B_NEC'].iloc[ind_gaps] = [np.full(3,np.nan)]*len(ind_gaps)
    df_out['B_NEC_CHAOS-all'].iloc[ind_gaps] = [np.full(3,np.nan)]*len(ind_gaps)    
    return df_out, df_ini.index[ind_gaps]


def res_param(res):
    '''
    Return sampling step and sampling frequency acording to
    Swarm magnetic data resolution

    Parameters
    --------
        res : string
            'HR' or 'LR'
    Returns
    -------
        sampling step (string) and sampling frequency (int)
    '''
    sstep = 'PT1S' if res=='LR' else 'PT0.019S'  # sampling step
    fs = 1 if res=='LR' else 50   # data sampling freq.
    return sstep, fs

def find_tshift2sat(dti, Rsph, sats):
    '''
    Compute orbital phase lag [in sec.] between two satellites
    
    Parameters
    ---------
    dti : Datetimeindex
        One second separated timestamps
    Rshp : numpy.array
        satellites position in spherical GEO frame
    sats : [str, str]
        satellite pair, e.g. [‘A’, ‘C’]
    
    Returns
    ------
    tshift : [float, float]
        optimal time shift in sec. to aligned the 
        two sensors side by side 
        
    '''
    ndti, nsc = len(dti), len(sats)
    i1, i2 = 0, np.min([ndti-2, 1300])      # indices used to select an orbit 
                                            # section (smaller that 1/4 orbit)
    deltat = (dti[i2] - dti[i1]).seconds    # duration of selected orbit section
    Rspheci = np.copy(Rsph)                 # will containe sats spherical 
                                            # coordinates in inertial frame
    Reci = np.full((ndti,nsc,3),np.nan)     # will containe sats cartesian 
                                            # coordinates in inertial frame
    # Spherical cordinates in the inertial frame are obtained from spherical
    # coordinates in GEO by compensating the longitude wit Earth rotation. Then
    # the cartesian coordinates in the inertial frame are found. The orbits' normals
    # 'neci' are found as the cross product of satellite positions at the start 
    # and stop of selected orbit section. The cross-product 'crossorb' of 
    # orbits' normal provides the direction of orbitat cross-point. Vector 
    # positions 'peci' of satellites at pseudo-equators (0 deg. latitude when 
    # considering the cross-point as pole) are computed. The evolution in time 
    # of the (signed) position angles in the plane peci, crossorb are considered
    # when computing the time shift
    neci, peci = (np.full((nsc,3),np.nan) for i in range(2))   # orbit normals and 
    theta, omega, tmean = (np.full(nsc,np.nan) for i in range(3))
    cosrun, sinrun, latrun = (np.full((nsc, i2-i1),np.nan) for i in range(3))
    for sc in range(nsc):
        Rspheci[:,sc,1] = (Rsph[:,sc,1] + np.arange(ndti)*360/(86164.098) +180)%360 -180
        Reci[:,sc,:], MATnec2eci_sc = R_in_GEOC(np.squeeze(Rspheci[:,sc,:]))
        neci[sc,:] = normvec(np.cross(Reci[i1,sc,:],Reci[i2,sc,:]))[0]
        theta[sc] = np.arccos(np.sum(normvec(Reci[i1,sc,:])*normvec(Reci[i2,sc,:])))
        omega[sc] = theta[sc]/deltat
    isclow = 0 if ((sats[0] == 'A') or (sats[0] == 'C')) else 1
    avelat = np.mean(Rspheci[i1:i2,isclow,0])
    vtmp = normvec(np.cross(neci[0,:],neci[1,:]))[0]
    crossorb = vtmp if vtmp[2]*avelat >0 else -vtmp         # cross-orbit unit vector 
    for sc in range(nsc):
        peci[sc,:] =  normvec(np.cross(crossorb, neci[sc,:]))[0]
        for ii in range(i1, i2):
            cosrun[sc,ii] = np.sum(normvec(Reci[ii,sc,:])*peci[sc,:])
            sinrun[sc,ii] = np.sum(normvec(Reci[ii,sc,:])*crossorb[:])
        latrun[sc,:] = np.arctan2(sinrun[sc,:], cosrun[sc,:])
        tmean[sc] = np.mean(latrun[sc,:] - np.pi/2)/omega[sc]
    tshift = [int(round(tmean[1] - tmean[0],0)), 0]
    print('Computed time shift array = ', tshift)
    return tshift

def split_into_sections(df, begend_arr):
    '''
    Splits the original DataFrame object df in more DataFrame
    objects according to time moments from begend_arr array.

    Parameters
    ---------
    df : DataFrame
        time indexed DataFrame object
    begend_arr : list
        time moments

    Returns
        list of DataFrame objects
    '''

    secorbs = []
    for start, end in zip(begend_arr[0:-1], begend_arr[1:]):
        secorb = df[start:end]
        secorbs.append(secorb)
    return secorbs

def find_ao_margins(df, fac_qnt = 'FAC_flt_sup', rez_qd = 100):
    '''
    Estimate the times of auroral oval (AO) encounter (central point 
    and edges) for a quarte-orbit section. Works with quasi-dipole 
    latitude QDLat (not time) dependence.
    
    Parameters
    ----------
    df : DataFrame
        Swarm data for a quarter-orbit section, including 
        FAC data 
    fac_qnt : string
        name of the column with FAC data to work with
    rez_qd : 
        resolution to interpolate the QDLat 
    Returns
    -------
    t64[0], t64[1], t64[2] : datetime-like 
        AO start, stop, and central time
    : float, float
        QDLat and QDLon of AO center
    qd_trend, qd_sign : float
        trend and sign of QDLat in the analysed quarte-orbit section
    ti_arr, qd_arr : numpy.array
        interpolated times and corresponding QDLat values   
    '''

    qd = df['QDLat'].values
    qdlon = df['QDLon'].values
    jb = df[fac_qnt].values
    ti = df['QDLat'].index.values.astype(float)
    qd_trend = (qd[-1] - qd[0])/abs(qd[-1] - qd[0])  # >0 if QDLat inreases
    qd_sign = qd[0]/abs(qd[0])  
    dqmin = np.round(np.amin(qd), decimals = 2)
    dqmax = np.round(np.amax(qd), decimals = 2) 
    nr = round((dqmax - dqmin)*rez_qd + 1)      
    qd_arr = np.linspace(dqmin, dqmax, nr)       # new QDLat array 
    if qd_trend > 0:
        ti_arr = np.interp(qd_arr, qd, ti)          # time of new QD points
        jb_arr = np.interp(qd_arr, qd, jb)          # FAC at new QD points
    else:
        ti_arr = np.interp(qd_arr, qd[::-1], ti[::-1])[::-1]
        jb_arr = np.interp(qd_arr, qd[::-1], jb[::-1])[::-1]
        qd_arr = qd_arr[::-1]
    if np.sum(np.abs(jb_arr)) > 0:
        jabs_csum = np.cumsum(np.abs(jb_arr))/np.sum(np.abs(jb_arr))
        idx_j1q = (np.abs(jabs_csum - 0.25)).argmin()
        idx_j2q = (np.abs(jabs_csum - 0.5)).argmin()
        idx_j3q = (np.abs(jabs_csum - 0.75)).argmin()
        idx_beg = np.max(np.array([0, idx_j1q - abs(idx_j3q - idx_j1q)]))
        idx_end = np.min(np.array([idx_j3q + abs(idx_j3q - idx_j1q), nr-1]))
        idx_ti_ao = (np.abs(ti - ti_arr[idx_j2q])).argmin()
        t64 = pd.DatetimeIndex(ti_arr[[idx_beg, idx_end, idx_j2q]]) 
        return t64[0], t64[1], t64[2], qd_arr[idx_j2q], qdlon[idx_ti_ao], \
                qd_trend, qd_sign, ti_arr, qd_arr  
    else:
        return pd.NaT, pd.NaT, pd.NaT, np.nan, np.nan, \
                qd_trend, qd_sign, ti_arr, qd_arr
    
def mva(v, cdir = None):
    '''
    Return results of Minimum Variance Analysis on an array of
    3D vector v. If cdir (3D vector) is provided, the analysis
    is performed in the plane perpendiculat to it

    Parameters
    ----------
    v : numpy.array
        3D vectors
    cdir : numpy.array
        direction to constrain MVA

    Returns
    -------
    array with eigenvalues and eigenvectors from MVA
    '''

    v_cov = np.cov(v, rowvar=False, bias=True)
    if cdir is not None:
        ccol = cdir.reshape(3,1)
        cunit = ccol / np.linalg.norm(ccol)
        d_mat = (np.identity(3) - np.dot(cunit,cunit.T))
        v_cov = np.matmul(np.matmul(d_mat, v_cov), d_mat)
    return np.linalg.eigh(v_cov)

def SortVertices(R4s, dB4s):
    '''
    Sort the quad's vertices in correct order; compute quad's
    parameters and the trace of the BI and LS reciprocal tensors.

    Parameters
    ----------
    R4s : numpy.array
        position vectors of the apexes in format [index, apex, comp]
    dB4s : numpy.array
        magnetic perturbation vectors at the apexes in format
        [index, apex, comp]

    Returns
    -------
    R4sa, dB4sa : numpy.arrays
        position and magnetic perturbation vectors sorted
        in correct order
    trBI : numpy.array
        trace of the reciprocal tensor in LS method
    Rmeso : numpy.array
        position of the mesocenter
    nuvec : numpy.array
        normal to the quad
    satsort : numpy.array
        satellite indeces in the correct order
    trLS : numpy.array
        trace of the reciprocal tensor in LS method
    EL, EM, el, em : numpy.arrays
        parameters of the four-point configuration
     '''

    Rmeso = np.mean(R4s, axis=-2)
    R4smc = R4s - Rmeso[:, np.newaxis, :]
    Rtens = np.sum(np.matmul(R4smc[:,:,:,None],R4smc[:,:,None, :]), axis = -3)
    eigval, eigvec = np.linalg.eigh(Rtens)
    # avoid eigenvectors flipping direction
    # for that, impose N to be closer to radial direction
    nprov = np.squeeze(eigvec[:,:,0])    # provisional normal
    cosRN = np.squeeze(np.matmul(Rmeso[...,None,:],nprov[...,:,None]))
    iposr = cosRN < 0
    eigvec[iposr,:,:] = -eigvec[iposr,:,:]
    # minimum variance direction is along normal N
    nuvec = np.squeeze(eigvec[:,:,0])    
    # find the rotation metrix from GEO to local proper frame
    Vcen = (R4s[:,2:,:] - R4s[:,0:2,:]).mean(axis=-2)
    yperp = np.cross(nuvec, Vcen)
    yunit = yperp/np.linalg.norm(yperp, axis=-1)[...,None]
    xperp = np.cross(yunit, nuvec)
    xunit = xperp/np.linalg.norm(xperp, axis=-1)[...,None]
    mgeo2pro = np.stack((xunit, yunit, nuvec), axis=-2)
    R4p, R4sa, dB4sa = (np.full_like(R4smc, np.nan) for i in range(3))
    # transform the sats positions in a local proper frame
    for jj in range(4):
        Rjj = R4smc[:,jj,:]
        R4p[:,jj,:] = np.matmul(mgeo2pro, Rjj[...,None]).reshape(Rjj.shape)
    # indexes to order the sats according to their position in the local proper
    # frame. New [0,1,2,3] indexes will correspond to quadrants Q3, Q4, Q1, and Q2     
    satsort = np.argsort(np.arctan2(R4p[:,:,1], R4p[:,:,0]), axis=- 1)  
    for jj in range(3):
        R4sa[:,:,jj] = np.take_along_axis(R4s[:,:,jj], satsort, -1)   
        dB4sa[:,:,jj] = np.take_along_axis(dB4s[:,:,jj], satsort, -1)        
        R4p[:,:,jj] = np.take_along_axis(R4p[:,:,jj], satsort, -1)
        
    # computes quad's parameters     
    EL = 0.5*np.abs(np.mean(R4p[:, 0:2, 1], axis = -1)) + \
            0.5*np.abs(np.mean(R4p[:, 2:, 1], axis = -1))
    EM = 0.25*np.abs(R4p[:, 1, 0] - R4p[:, 0, 0]) + \
            0.25*np.abs(R4p[:, 2, 0] - R4p[:, 3, 0])
    el = 0.5*np.abs(np.mean(R4p[:, 0:2, 0], axis = -1)) + \
            0.5*np.abs(np.mean(R4p[:, 2:, 0], axis = -1))
    em = 0.25*np.abs(R4p[:, 1, 1] - R4p[:, 0, 1]) + \
            0.25*np.abs(R4p[:, 2, 1] - R4p[:, 3, 1])
    # computes the trace of the BI and LS reciprocal tensors
    D = 16*((EL*EM)**2 + (EM*em)**2 + (el*em)**2)
    trLS = 4/D*(EM**2 + el**2 + EL**2 + em**2)
    trBI = 0.25/((EL*EM)**2)*(EM**2 + EL**2 + el**2)
    return R4sa, dB4sa, trBI, Rmeso, nuvec, satsort, trLS, EL, EM, el, em
