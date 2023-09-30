# -*- coding: utf-8 -*-
"""
This document exemplifies the use of the ETFRotationalMomentumBackTest class
to backtest rotational momentum trading with nondefault metric settings and
with user defined functions.
"""

import pandas as pd
from ETFRotationalMomentum import ETFRotationalMomentumBackTest
from VRatioScorer import compute_vratio_score


#Example of using custom metrics with the vratio function


#Load data
dfP = pd.read_csv('21_NoSHY.csv', parse_dates=['Date'])
dfAP = pd.read_csv('21_NoSHY_AP.csv', parse_dates=['Date'])


dfP_list = [dfP]
dfAP_list = [dfAP]

#Have to sort values by date before setting index otherwise generate error 
#When calculating returns ...............!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

for idx in range(len(dfP_list)):
    
    dfP_list[idx] = dfP_list[idx].sort_values(by='Date')
    dfAP_list[idx] = dfAP_list[idx].sort_values(by='Date')
    
    dfP_list[idx].set_index('Date', inplace = True)
    dfAP_list[idx].set_index('Date', inplace = True)
    
    dfP_list[idx] = dfP_list[idx].loc["2021-01-01":"2022-01-01"]
    dfAP_list[idx] = dfAP_list[idx].loc["2021-01-01":"2022-01-01"]
    



portfolio_weights_list = [1]




#Setup Custom metrics

#Step 1: Add custom function to dictionary and clarify if a lower value should
#be scored higher or not
#We set the vratio with the lower_is_better set to false (to trade momentum)
custom_metric_func_dict = {"v": (compute_vratio_score, False)}


#Step 2: setup metrics specifics to be used
#Need the key for the function in the dictionary plus all arguments (except dfP)
#Here using 20 day (short term) momentum, 66 day(long term) momentum
#short term volatility and 2 day lag v-ratio with a window size of 200.
metrics = [("m",20),("m",66),("s",20),("v",2,200)] 


#Step 3: setup the weights for each metric
weights =  [0.25, 0.1, 0.4, 0.25]


#Step 4: Instantiate the tester and pass in the the parameters from previous
#steps. No laddering is being used at the moment
laddering = False
max_risk_holding = 3
tester = ETFRotationalMomentumBackTest(lookback = 20, holding_freq ="11W-FRI",
                                       custom_metric_func_dict=custom_metric_func_dict,
                                       metrics = metrics, weights = weights,
                                       max_risk_holding = max_risk_holding)

#Step 5: Pass the data and Run it
dfPRR_test = tester.backtest_portfolios(dfP_list, dfAP_list, portfolio_weights_list, laddering = laddering, plotting = False)