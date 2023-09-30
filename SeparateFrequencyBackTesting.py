# -*- coding: utf-8 -*-
"""
This document exemplifies the use of the ETFRotationalMomentumBackTest class
to backtest rotational momentum trading of ETF sets with their own trading 
frequency.
"""

import pandas as pd
from ETFRotationalMomentum import ETFRotationalMomentumBackTest




#Load Data
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
    
    dfP_list[idx] = dfP_list[idx].loc["2017-01-01":"2018-01-01"]
    dfAP_list[idx] = dfAP_list[idx].loc["2017-01-01":"2018-01-01"]


portfolio_weights_list = [0.5,0.5]





#populate frequency list according to requirements
lookback_list = [20,30]
frequency_list = ["11W-FRI", "16W-FRI"]
laddering = False
max_risk_holding = 3

dfPRR_list = []
for portfolio_index in range(len(dfP_list)):
    
    tester = ETFRotationalMomentumBackTest( lookback = lookback_list[portfolio_index],
                                           holding_freq = frequency_list[portfolio_index],
                                           max_risk_holding = max_risk_holding)
    #Use backtest, not backtest_portfolios
    dfPRR = tester.backtest(dfP_list[portfolio_index],
                                    dfAP_list[portfolio_index],
                                    laddering = laddering,
                                    plotting = False,
                                    show_trailing = False)
    dfPRR_list.append(dfPRR)
    
    dfPRR.to_csv(r'Results\dfPRR_set_' + str(portfolio_index)+'.csv', header = True, index=True, encoding='utf-8')


#Combine results of the runs
dfPRR_combined = pd.DataFrame(index = dfPRR_list[0].index)

dfPRR_combined["ALL_R"] = sum(dfPRR_list[i]["ALL_R"] * portfolio_weights_list[i] for i in range(len(dfPRR_list)) )
dfPRR_combined["DETREND_ALL_R"] = sum(dfPRR_list[i]["DETREND_ALL_R"] * portfolio_weights_list[i] for i in range(len(dfPRR_list)) )
dfPRR_combined["I"] = sum(dfPRR_list[i]["I"] * portfolio_weights_list[i] for i in range(len(dfPRR_list)) )

print("Combined Portfolio Results \n")

tester.compute_financial_metric_results( dfPRR_combined, trailing_stop=False, print_metrics=True)
tester.compute_financial_metric_results( dfPRR_combined, trailing_stop=True, plotting = True, print_metrics=True)


