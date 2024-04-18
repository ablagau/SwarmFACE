#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from viresclient import SwarmRequest

def single(dtime_beg, dtime_end, sat):
    '''
    Retrieve the Level-2 single-satellite FAC density from the
    ESA database

    Parameters
    ----------
    dtime_beg : str
        start time in ISO format 'YYYY-MM-DDThh:mm:ss'
    dtime_end : str
        end time in ISO format
    sat : [str]
        satellite, e.g. ['A']

    Returns
    -------
    FAC_L2 : DataFrame
        the Level-2 single-satellite FAC and IRC densities
    '''

    request = SwarmRequest()
    request.set_collection('SW_OPER_FAC'+sat[0]+'TMS_2F')
    request.set_products(measurements=["FAC","IRC"], sampling_step="PT1S")
    data = request.get_between(start_time = dtime_beg, 
                               end_time = dtime_end,
                               asynchronous=False)  
    print('Used FAC file: ', data.sources[0])
    FAC_L2 = data.as_dataframe()
    return FAC_L2

def dual(dtime_beg, dtime_end):
    '''
    Retrieve the Level-2 dual-satellite FAC density from the
    ESA database

    Parameters
    ----------
    dtime_beg : str
        start time in ISO format 'YYYY-MM-DDThh:mm:ss'
    dtime_end : str
        end time in ISO format

    Returns
    -------
    FAC_L2 : DataFrame
        the Level-2 dual-satellite FAC and IRC densities
    '''

    request = SwarmRequest()
    request.set_collection('SW_OPER_FAC_TMS_2F')
    request.set_products(measurements=["FAC","IRC"], sampling_step="PT1S")
    data = request.get_between(start_time = dtime_beg, 
                               end_time = dtime_end,
                               asynchronous=False)  
    print('Used FAC file: ', data.sources[0])
    FAC_L2 = data.as_dataframe()
    return FAC_L2
