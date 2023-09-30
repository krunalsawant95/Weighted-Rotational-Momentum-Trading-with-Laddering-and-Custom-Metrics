# -*- coding: utf-8 -*-

"""
This document exemplifies the use of the ETFRotationalMomentumBackTest class
to backtest rotational momentum trading of ETF sets with the same set of 
metrics and trading frequency through the use of its backtest_portfolios 
method.
"""
import pandas as pd
from ETFRotationalMomentum import ETFRotationalMomentumBackTest

#Backtesting with the same frequency and lookback
#Synchronized Frequency BackTesting


dfP = pd.read_csv('21_NoSHY.csv', parse_dates=['Date'])
dfAP = pd.read_csv('21_NoSHY_AP.csv', parse_dates=['Date'])

dfP2 = pd.read_csv('7USD_Bonds.csv', parse_dates=['Date'])
dfAP2 = pd.read_csv('7USD_Bonds_AP.csv', parse_dates=['Date'])

dfP_list = [dfP,dfP2]
dfAP_list = [dfAP,dfAP2]

#Have to sort values by date before setting index otherwise generate error 
#When calculating returns ...............!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

for idx in range(len(dfP_list)):
    
    dfP_list[idx] = dfP_list[idx].sort_values(by='Date')
    dfAP_list[idx] = dfAP_list[idx].sort_values(by='Date')
    
    dfP_list[idx].set_index('Date', inplace = True)
    dfAP_list[idx].set_index('Date', inplace = True)
    
    dfP_list[idx] = dfP_list[idx].loc["2021-01-01":"2022-01-01"]
    dfAP_list[idx] = dfAP_list[idx].loc["2021-01-01":"2022-01-01"]


portfolio_weights_list = [0.5,0.5]


tester = ETFRotationalMomentumBackTest(lookback = 20, holding_freq ="11W-FRI", max_risk_holding = 3)

# laddering = False
laddering = True

dfPRR_test = tester.backtest_portfolios(dfP_list, dfAP_list, portfolio_weights_list, laddering = laddering)