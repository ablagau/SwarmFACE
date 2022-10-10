#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 21:06:18 2022

@author: blagau
"""
import pandas as pd
from viresclient import set_token
from viresclient import SwarmRequest

def single(dtime_beg, dtime_end, sat):
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
    request = SwarmRequest()
    request.set_collection('SW_OPER_FAC_TMS_2F')
    request.set_products(measurements=["FAC","IRC"], sampling_step="PT1S")
    data = request.get_between(start_time = dtime_beg, 
                               end_time = dtime_end,
                               asynchronous=False)  
    print('Used FAC file: ', data.sources[0])
    FAC_L2 = data.as_dataframe()
    return FAC_L2
