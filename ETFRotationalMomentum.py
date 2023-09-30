import pandas as pd
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import detrendPrice
import WhiteRealityCheckFor1
import re
import itertools

class ETFRotationalMomentumBackTest:
    """
   
    
    Class that implements an adaptation of the ETF rotational momentum 
    backtesting strategy code developed by APS1051 Instructors Sabatino 
    Constanzo-Alvarez and Rosario Trigo-Ferre.
    
    Initialization requires a minimum lookback period and optionally, metrics 
    to be used for scoring and ranking ETFs, positive weights for each metric
    summing up to 1 and a specified trading frequency according to 
    pandas.asfreq.
    
    If no metrics are given then default metrics are used - short term 
    momentum and volatility based on the lookback period, and a long term
    momentum.
    
    """
    
    def __init__(
            self, lookback, holding_freq, delay = 1, metrics = None,
            weights = None, custom_metric_func_dict = None,
            max_risk_holding = 3):
        """
        

        Parameters
        ----------
        lookback : int
            Short term period over which returns/momentum is calculated
        
        freq : string, optional
             Trading frequency for resampling accoring to pandas.asfreq
             The default is "11W-FRI".
        
        delay : int, optional
             Delay between signal and trade initialization. The default is 1.
        
        custom_metric_func_dict: dictionary whose values are (metric_function, lower_is_better)
            Defines the metric functions to be used and whether a lower score is 
            a better score or not. By default
            metric_func_dict with defaults:
                'm': momentum of returns, lower_is_better = False
                's': standard deviation, lower_is_better = True
            custom_metric_func_dict updates metric_func_dict, overriding any
            existing defaults.
            Together with metrics this allows for the combination of any 
            user-defined metric function and the default metrics.
            
            Note default functions use trading days
        
        metrics : list of tuples in the form (string, args), optional
            A list of metrics and the duration over which they are to be
            calculated . The string is the key for a metric function in
            custom_metric_func_dict with defaults:
                'm' - momentum of returns
                's' - standard deviation
            The args are the arguments to the function, excluding the price df.
            This class by default assumes a scorer uses the price df (dfP) as
            the first argument. Eg. passing in [("m",t1)] results in a 
            function call: compute_momentum(dfP,t1)
            Passin in an [("f",t1, t2, t3)] results in a function call:
                myfunc(dfP,t1, t2, t3) for computing scores once myfunc
                has been added to the dictionary with the key 'f'
            
            If no metrics are passed a default of  [("m",t1),("m",t2),("s",t1)] 
            is used where t1 is the lookback given and t2 is a longer term 
            lookback = 3*t1+((3*t1)//20)*2 
        
        weights : list of weights to apply to each metric. If no metrics are 
            given but three weights are given the weights are used for the
            default metrics. Otherwise weights must match metrics given.

        """
        
        self.lookback = lookback
        self.freq = holding_freq
        self.delay = delay
        self.default_weights = [0.3, 0.3, 0.4]
               
        if (metrics is None):
            
            #If no metrics are given calculate default metrics based on the 
            #lookback given
            t1 = int(lookback)
            t2 = 3*t1+((3*t1)//20)*2 
            self.metrics = [("m",t1),("m",t2),("s",t1)] #List of tuples where each tuple is (metric type, args)
            
            
            if (weights is None):
                #If we are not given weights use the default weights
                self.weights = self.default_weights
            else:
                #If we are given weights they must match the default metrics
                
                statement = "If no metrics are given, the weights given must" \
                    + " have size 3 in order to be used with default metrics."
                assert (len(weights) == 3), statement
                self.weights = weights
             
        else:
            self.metrics = metrics
            assert len(weights) == len(metrics), "Weights must have the same length as metrics given."
            self.weights = weights
        
        
        
        self.metric_func_dict = {"m": (self.compute_momentum, False) , "s": (self.compute_volatility, True)}
        
        if custom_metric_func_dict is not None:
            self.metric_func_dict.update(custom_metric_func_dict) #Expected as dict of tuples of key:(function,LowerIsBetter)
        
        #Limit the max risk holding to not go beyond the limit of our
        #trading frequency
        
        self.holding_len = re.findall(r"[\d]+", self.freq) #Get digits
        self.holding_len = int(self.holding_len[0]) #Convert digits to int
        
        if max_risk_holding <= self.holding_len:
            self.max_risk_holding = max_risk_holding
        else:
            self.max_risk_holding = self.holding_len
        return


    def get_date(self, dt):
        """
        Grabs datetime object from string date
        
        Parameters
        ----------
        dt : string containing date information

        Returns
        -------
        datetime_object: a datetime.datetime object containing the date
        retrieved from dt

        """
        if type(dt) != str:
            return dt
        try:
            datetime_object = datetime.datetime.strptime(dt, '%Y-%m-%d')
        except Exception:
            datetime_object = datetime.datetime.strptime(dt, '%m/%d/%Y')
            return datetime_object
        else:
            return datetime_object

    def detrend_prices(self, dfAP):
        
        
        dfDetrend = pd.DataFrame(np.zeros_like(dfAP.values), index = dfAP.index,
                                 columns = dfAP.columns)
        
        for column in dfAP.columns:
            dfDetrend[column] =  detrendPrice.detrendPrice( dfAP.loc[:, column] ).values
        
        return dfDetrend

    def compute_momentum(self,dfP, periods):
        """

        Parameters
        ----------
        dfP : pandas dataframe of stock prices with a datetime index
        
        periods : int, length of lookback period over which to compute returns in business days

        Returns
        -------
        momentum_df:
            pandas dataframe of the momentum with the same shape as dfP.
            The nans generated in the returns are filled by padding. 

        """
        
        momentum_df = dfP.pct_change(periods = periods-1, fill_method='pad', limit=None, freq=None)
        return momentum_df


    def compute_volatility(self,dfP, periods):
        """

        Parameters
        ----------
        dfP : pandas dataframe of stock prices with a datetime index
        
        periods : int, length of lookback period over which to compute returns in business days

        Returns
        -------
        momentum_df:
            pandas dataframe of the standard deviation with the same shape as dfP.
            The nans generated in the returns are filled by padding. 

        """
        volatility_df = pd.DataFrame(np.zeros_like(dfP.values),
                                index = dfP.index, columns = dfP.columns)
        
        #Compute one date returns 
        dfR = dfP.pct_change(periods=1, fill_method='pad', limit=None, freq=None)
        
        #Compute volatility of returns for each stock
        columns = dfP.shape[1]
        for column in range(columns):
            volatility_df[dfP.columns[column]] = (dfR[dfP.columns[column]].rolling(window = periods).std())*math.sqrt(252)
        
        return volatility_df

    def compute_ranks(self,scores, lower_is_better = False):
        """
        Takes in a set of scores generated by a metric function and the 
        criterion whether lower scores are better (as for volatility). Then
        returns the rankings according to the scores and criterion.

        Parameters
        ----------
        scores : 
            dataframe of metric scores for the stock prices
  
        lower_is_better : boolean, optional
            Whether decreasing values are scored higher or lower.The default 
            is False, in which case, higher values receive a higher ranking.

        Returns
        -------
        ranks_df : 
            dataframe of the same shape as scores, containing the rank for
            each stock at every date


        """
        ranks_df = pd.DataFrame(np.zeros_like(scores.values),
                                index = scores.index, columns = scores.columns)
        
        for row in range(scores.shape[0]):
            arr_row = scores.iloc[row].values
            if lower_is_better == False:
                temp = arr_row.argsort() #Best is ETF with largest score
            else:
                temp = (-arr_row).argsort()[:arr_row.size] #Best is ETF with lowest score
           
            #Lowest score is rank 1, best score is rank(num_ETF's)
            ranks_df.iloc[row,temp] = np.arange(1,len(arr_row)+1)
            

        
        return ranks_df

    def generate_weighted_rankings(self, dfP):
        """
        Calls each score function on the prices dataframe and converts them to
        ranks based on whether a lower score is better (volatility) or not.
        Then uses the metric weights in order to compute the overall rankings

        Parameters
        ----------
        dfP : Dataframe of prices

        Returns
        -------
        score_df : Dataframe of the final rankings for each ETF in dfP

        """
        
        #Create a dataframe to hold all the rankings and weigh them
        score_df = pd.DataFrame(np.zeros_like(dfP.values), index = dfP.index,
                                columns = dfP.columns)
        
        for index, metric in enumerate(self.metrics):
            type = metric[0]
            #period = metric[1]
            
            
            #compute the metric
            metric_func, lower_is_better = self.metric_func_dict[type] #retrieve appropriate function
            
            
            metric_score = self.compute_ranks(metric_func(dfP, *metric[1:]), lower_is_better)
            weighted_metric_score = self.weights[index] * metric_score #multiply by weights
            
            #Add the metric to the current overall score rankings
            score_df = score_df.add(weighted_metric_score, fill_value = 0)
            
        return score_df
        pass
    
    def select_ETF(self, dfP):
        """
        Calls the ranking functions and selects the ETF with the highest rank. 
        Then, updates the dataframe of choices to 1 on the day the ETF is selected
        
        Returns 
        -------
        dfChoice: dataframe of choices.
        """

    
        
        dfAll_ranks = self.generate_weighted_rankings( dfP)
        
        new_columns = [column + "_CHOICE" for column in dfP.columns]
        
        dfChoice = pd.DataFrame(np.zeros_like(dfP.values), index = dfP.index,
                                columns = new_columns)
        
        rows = dfChoice.shape[0]


        #this loop takes each row of the All-ranks dataframe, puts the row into an array, 
        #within the array the contents scanned for the maximum element 
        #then the maximum element is placed into the Choice dataframe
        for row in range(rows):
            arr_row = dfAll_ranks.iloc[row].values
            max_arr_column = np.argmax(arr_row, axis=0) #gets the INDEX of the max
            dfChoice.iat[row, max_arr_column] = 1
        
        return dfChoice
    
    def select_ETF_with_laddering(self, dfP):
        """
        Selects ETFs according to a laddering paradigm where up to 
        freq/max_holding ETFs may be selected at a time.

        Parameters
        ----------
        dfP : pandas dataframe of prices.

        Returns
        -------
        dfChoice : pandas dataframe of the same shape as dfP with the same 
            index and new columns [dfP.Column_Choice].This holds 1 where an
            ETF is selected and 0 where it is not
        slot_trade_dates_list : List of datetimes when a trade can occur
        dfAll_ranks_copy : Tpandas dataframe of the rankings of each ETF,
            modified to -np.inf where that ETF is already slotted.

        """
        
    
        dfAll_ranks = self.generate_weighted_rankings(dfP)
        
        new_columns = [column + "_CHOICE" for column in dfP.columns]
        
        dfChoice = pd.DataFrame(np.zeros_like(dfP.values), index = dfP.index,
                                columns = new_columns)
        
        rows = dfChoice.shape[0]
    
        #######################################
        #determine number of partions over time
        
        #Assume input in risk holding period is same type (Week/day) as holding_freq
        holding_len = re.findall(r"[\d]+", self.freq) #Get digits
        holding_len = int(holding_len[0]) #Convert digits to int
        
        #remainder is everything after holding period digits
        remainder = re.findall( str(holding_len) + r"(.*)", self.freq)
        remainder = remainder[0] #Take the string result out of the list
        
        max_risk_holding = self.max_risk_holding
        
        """
        In the ladder approach since we cannot risk investing our wealth more
        than max_risk_holding periods, but we need to hold the ETF for the 
        holding/frequency periods in order to develop the full momentum, we 
        instead invest a portion of our wealth for the full duration to 
        reduce risk. The proportion that shall be invested is holding periods
        divided by max_risk_holding periods. We then adjust another proportion
        of our wealth after 1 max_risk_holding_periods...
        
        """
        
        num_ladder_partitions = math.ceil(holding_len/max_risk_holding) #Available trading bins
        num_full_partitions = (holding_len//max_risk_holding) #Number of trading bins that are max_risk_holding long
        final_partition_size = holding_len - num_full_partitions * max_risk_holding
        
        #Get the first day we can actually trade on, according to our frequency.
        #We will modify our choices from there on
        trade_start = dfP.asfreq(self.freq).index[0]
        time_delta = dfP.asfreq(self.freq).index[1] - dfP.asfreq(self.freq).index[0]
        
        #Need to generate the dates we can trade on for each bin
        #Resample every holding period and every max_risk_holding period
        #starting from trade_start then combine the lists
        #Only necessary if holding period is indivisible by max risk holding
        dates_df = pd.DataFrame(dfChoice.index, index = dfChoice.index)
        dates_df = dates_df.loc[trade_start:]
        dates_1 = dates_df.asfreq(self.freq).index
        dates_2 = dates_df.asfreq(str(max_risk_holding) + remainder).index
        
        #Get all start dates for bins
        slot1_dates = dates_1.union(dates_2)
        slot_start_dates = slot1_dates[:num_ladder_partitions] 
        
        slot_trade_dates_list = []
        
        #Determine slot choice dates for other slots
        for index in range(num_ladder_partitions):
            slot_trade_start = slot_start_dates[index]
            dates_df = dates_df.loc[slot_trade_start:]
            current_slot_trade_dates = dates_df.asfreq(self.freq).index
            
            #only use a date if its in the original index
            #otherwise get the closest date
            
            #Fixed dates are dates in our index
            #As we may otherwise set trades on holidays 
            #In this case we may slightly violate our max risk holding by 1 day
            #or undervalue our momentum by 1 day
            fixed_slot_trade_dates = [date if date in dfChoice.index \
                                        else dfChoice.index[ dfChoice.index.searchsorted(date) ] \
                                            for date in current_slot_trade_dates ]
            fixed_slot_trade_dates = pd.DatetimeIndex(fixed_slot_trade_dates)
            
            
            if index == 0:
                slot_trade_dates = fixed_slot_trade_dates
            else:
                slot_trade_dates = slot_trade_dates.union(fixed_slot_trade_dates)
                
            
            slot_trade_dates_list.append(fixed_slot_trade_dates) #Used for matching in backtest
            
        #Now we repeat the matching exercise for the backtest as before but manipulate the ranks df.
        
        #To assist with the determination of Long entries and exists we will
        #Modify the ranks as we make choice selections, so the ranks reflect availability.
        dfAll_ranks_copy = dfAll_ranks.copy(deep = True)
        
        
        #Generate entry
        #Choices are 0 except on slot trade dates
        trade_rows = len(slot_trade_dates)
        for row in range(trade_rows):
            #Note our row and the end row to which we are filling in values
            start = slot_trade_dates[row]
            
            if row <= trade_rows - num_ladder_partitions :
                end = slot_trade_dates[row + num_ladder_partitions-1]
            else:
                end = dfAll_ranks_copy.index[-1]
             
            #Need to check alignment here
            arr_row = dfAll_ranks_copy.loc[start].values
            max_arr_column = np.argmax(arr_row, axis=0) #gets the INDEX of the max
            target_choice_column = dfChoice.columns[max_arr_column] 
            dfChoice.loc[slot_trade_dates[row] , target_choice_column] = 1
        
            #Then change scores in the ranking df to be negative infinity so they 
            #not considered until the ETF chosen is up for reselection
            target_column = dfAll_ranks_copy.columns[max_arr_column]
            
            #Record the stard and end rank values 
            val = np.max(arr_row, axis=0)
            end_val =  dfAll_ranks_copy.loc[end, target_column]
            
            #Filling in the middle values
            #Because start and end are from the slot trade dates and not the
            #index we will overwrite them and then replace them with their 
            #original values from before.
            if row != trade_rows - 1:
                dfAll_ranks_copy.loc[start:end, target_column] = -np.inf
            else:
                dfAll_ranks_copy.loc[start:, target_column] = -np.inf
            
            #Replace the start and end values to consider the case where
            #we maintain the position.
            dfAll_ranks_copy.loc[start, target_column] = val
            dfAll_ranks_copy.loc[end, target_column] = end_val
        
        #Replace 
        dfAll_ranks_copy = dfAll_ranks_copy.replace(to_replace= -np.inf, value = np.nan)
      
        
    
        return  dfChoice, slot_trade_dates_list, dfAll_ranks_copy


    
    def backtest(self, dfP, dfAP, laddering = False, show_trailing = True, return_metrics = False, trailing_stop=False, plotting = False, print_metrics = False):
        """
        Backtests an individual portfolio/set of ETFs according to the 
        rotational momentum strategy.

        Parameters
        ----------
        dfP : pandas dataframe of prices, used for calculating signals
        
        dfAP : pandas dataframe of adjusted prices, used for calculating 
            metrics.
            
        laddering : Boolean, optional
            Whether to apply laddering in the ETF selection. The default is 
            False.
            
        show_trailing : Boolean, optional
            When show_trailing is true and trailing_stop is false. The plot of
            the equity curve and the filtered equity curve with trailing stop 
            will be plotted. The default is True.
            
        return_metrics : Boolean, optional
            Whether the metrics computed using the given trailing stop option 
            should be returned or not. If this is true the functions returns
            dfPRR - the dataframe of the strategies returns using the adjusted
            prices and metrics - a tuple of metrics. Otherwise only dfPRR is
            returned. The default is False.
            
        trailing_stop : Boolean, optional
            Whether to use a trailing stop or not in computing the financial
            metrics. The default is False.
            
        plotting : Boolean, optional
            Boolean that determines whether the equity curve is plotted. If 
            show_trailing is True and trailing_stop is False an equity curve
            will still be plotted to show the difference in using a trailing
            stop vs otherwise. The default is False.
            
        print_metrics : Boolean, optional
            If True, prints the metrics of the selected trailing stop option.
            Note that is show_trailing is True trailing stop metrics will still
            be printed out. The default is False.

        Returns
        -------
        dfPRR: pandas dataframe of returns according to the adjusted prices,
            supplemented with equity curve.
        
        metrics: Optional, only returned if return_metrics is True. 
            Tuple containing:
            trading days based total return
            total return
            trading days based CAGR
            CAGR
            Sharpe Ratio
            Volatility

        """
        
        assert all(dfP.columns == dfAP.columns), "The columns of the ETF prices and adjusted prices dataframes must match."
        
        dfDetrend = self.detrend_prices(dfAP)
        frequency = self.freq
        
        dfPRR= dfAP.pct_change()
        dfDetrendRR = dfDetrend.pct_change()
 
        
        
        if laddering == False:
            dfChoice = self.select_ETF(dfP)
            #Make note of the choices
            dfPRR = pd.concat( (dfPRR, dfChoice), axis = 1, join='outer')
            
            #dfT is the dataframe where the trading day is calculated. 
            dfT = dfP.drop(labels=None, axis=1, columns=dfP.columns)
            
            dfT['DateCopy'] = dfT.index
            dfT1 = dfT.asfreq(freq= frequency, method='pad')
            dfT1.set_index('DateCopy', inplace=True)
            dfTJoin = pd.merge(dfT,
                             dfT1,
                             left_index = True,
                             right_index = True,
                             how='outer', 
                             indicator=True)
    
            dfTJoin = dfTJoin.loc[~dfTJoin.index.duplicated(keep='first')] #eliminates a row with a duplicate index which arises when using kibot data
            
            dfPRR=pd.merge(dfPRR,dfTJoin, left_index=True, right_index=True, how="inner")
            dfPRR.rename(columns={"_merge": frequency +"_FREQ"}, inplace=True)
        else:
            dfChoice, slot_trade_dates_list, ranks_available = self.select_ETF_with_laddering(dfP)
            #Make note of the choices
            dfPRR = pd.concat( (dfPRR, dfChoice), axis = 1, join='outer')
            num_slots = len(slot_trade_dates_list)
            dfT_columns = ["slot_" + str(slot) for slot in range(num_slots)]
            dfT_columns = dfT_columns + ["Dates"]
            
            dfTJoin = pd.DataFrame(np.zeros(shape = (dfPRR.shape[0], num_slots+1)),
                                        index = dfPRR.index,
                                        columns =  dfT_columns)
            dfTJoin["Dates"] = dfTJoin.index.values
            
            for slot_index in range(num_slots):
                #Determine where dates match
                slot_bucket = slot_trade_dates_list[slot_index]
                slot_column = dfTJoin.columns[slot_index]
                dfTJoin.loc[dfTJoin["Dates"].isin(slot_bucket), slot_column] = "Both"
            
            dfPRR=pd.merge(dfPRR,dfTJoin, left_index=True, right_index=True, how="inner")
            
            #Condenses all slots together to generate Boolean series
            #This combined with availability will be used to determine entry and exits
            dfTtemp = dfTJoin.drop(["Dates"], axis = 1).replace(to_replace=0, value = False)
            
    
    

        ########################################
        #Generating long entry, long exit, num long and shifted return values
        
        #Generate the list of column names
        temp_columns = [[column+"_LEN", column+"_LEX", column+"_NUL", column + "_RET" ]  for column in dfP.columns]
        new_columns = list(itertools.chain.from_iterable(temp_columns))
        
        #Generate an empty array that fits the list of column names
        #and will have the same rows as the results dataframe
        entry_exit_array = np.zeros(shape = (dfPRR.shape[0], len(new_columns)) )
        entry_exit_df = pd.DataFrame(entry_exit_array, index = dfP.index,
                                     columns = new_columns)
        
        #Combine with the results dataframe
        dfPRR = pd.concat( (dfPRR, entry_exit_df), axis = 1, join='outer')
        
        
        
        ########################################
        #Fill the entry and exit positions for each ETF
        Frequency = self.freq
        for column in dfP.columns:
            choice_col = column + "_CHOICE"
            len_col = column + "_LEN"
            lex_col = column + "_LEX"
            nul_col = column + "_NUL"
            ret_col = column + "_RET"
            
    
            
            if laddering == False:
                dfPRR.loc[:, len_col] = ((dfPRR[Frequency+"_FREQ"] =="both") & (dfPRR[choice_col] ==1))
                dfPRR.loc[:, lex_col] = ((dfPRR[Frequency+"_FREQ"] =="both") & (dfPRR[choice_col] !=1))
            
            else:
                dfPRR.loc[:, len_col] = ((dfTtemp.any(axis = 1)) & (dfPRR[choice_col] ==1) & (~ranks_available[column].isna()))
                dfPRR.loc[:, lex_col] = ((dfTtemp.any(axis = 1)) & (dfPRR[choice_col] !=1) & (~ranks_available[column].isna()) )
            
            
            

            #use padding to fill in the number of units long (NUL) column
            dfPRR.loc[:, nul_col] = np.nan
            
            dfPRR.loc[dfPRR[lex_col] == True, nul_col ] = 0
            if laddering == False:
                dfPRR.loc[dfPRR[len_col] == True, nul_col ] = 1
            else:
                dfPRR.loc[dfPRR[len_col] == True, nul_col ] = (1/num_slots) #this order is important
            
            #By default the first long entry is not one from the strategy
            #but random or the SHY. This code removes it and propogates 0
            #via padding until the first true long entry from the strategy.
            dfPRR.iat[0,dfPRR.columns.get_loc(nul_col)] = 0
            dfPRR[nul_col] = dfPRR[nul_col].fillna(method='pad')
            
        
            #Compute the shifted returns (the returns we would get from this strategy)
            dfPRR[ret_col] = dfPRR[column]*dfPRR[nul_col].shift(self.delay)
          
        
        
        
        ########################################
        #Generate the shifted return values from the strategy for the detrended prices
        new_columns = [column + "_RET"  for column in dfP.columns]
        temp_df = pd.DataFrame(np.zeros_like(dfP.values), index = dfP.index,
                               columns = new_columns)
        
        dfDetrendRR = pd.concat( (dfDetrendRR, temp_df), axis = 1, join='outer')
        
        for column in dfP.columns:
            nul_col = column + "_NUL"
            ret_col = column + "_RET"
            #repeat for detrended returns
            dfDetrendRR[ret_col] = dfDetrendRR[column]*dfPRR[nul_col].shift(self.delay)
        
        
        ########################################
        #Compute the portfolio returns... 
        #Needs to be adjusted for laddering, by considering total num_units_long
        
        #compute all returns
        dfPRR["ALL_R"] = 0
        dfDetrendRR["ALL_R"] = 0
        
        for column in dfP.columns:
            ret_col = column + "_RET"
            dfPRR["ALL_R"] = dfPRR["ALL_R"] + dfPRR[ret_col]
            #repeat for detrended returns
            dfDetrendRR["ALL_R"] = dfDetrendRR["ALL_R"] + dfDetrendRR[ret_col]
        
        dfPRR["DETREND_ALL_R"] = dfDetrendRR["ALL_R"]
        
        #Convert all percent returns to cumulative returns
        dfPRR = dfPRR.assign(I =np.cumprod(1+dfPRR['ALL_R'])) #this is good for pct return
        dfPRR.iat[0,dfPRR.columns.get_loc('I')]= 1
        #repeat for detrended returns
        dfDetrendRR = dfDetrendRR.assign(I =np.cumprod(1+dfDetrendRR['ALL_R'])) #this is good for pct return
        dfDetrendRR.iat[0,dfDetrendRR.columns.get_loc('I')]= 1

        dfPRR = dfPRR.assign(DETREND_I = dfDetrendRR['I'])
        
        metrics = self.compute_financial_metric_results(dfPRR, trailing_stop=trailing_stop, plotting = plotting, return_metrics = return_metrics, print_metrics = print_metrics)
        
        if (show_trailing == True) and (trailing_stop == False):
            #Compute and plot trailing stop graph, but do not return metrics
            self.compute_financial_metric_results(dfPRR, trailing_stop= True, plotting = True, print_metrics = True)
        
        if return_metrics == False:
            return dfPRR
        else:
            return dfPRR, metrics
    

    def backtest_portfolios(self, dfP_list, dfAP_list, portfolio_weights_list, laddering = False, trailing_stop = False, show_trailing = True,  return_metrics = False, plotting = False, print_metrics = False):
        """
        Calls backtest to apply rotational momentum strategy multiple times on 
        each individual set of ETFs in a list of sets that constitue a 
        portfolio. Combines the results into a single

        Parameters
        ----------
        dfP_list : list of pandas dataframes of prices for each ETF set. Used
            for calculating signals
            
        dfAP_list : list of pandas dataframes of adjusted prices for each ETF 
            set. Used for calculating metrics. The sets must be in the same 
            order as in dfP_list
            
        portfolio_weights_list: list of weights which to apply to each 
            individual ETF set when combining the results into a single 
            portfolio.
            
        laddering : Boolean, optional
            Whether to apply laddering in the ETF selection. The default is 
            False.
            
        show_trailing : Boolean, optional
            When show_trailing is true and trailing_stop is false. The plot of
            the equity curve and the filtered equity curve with trailing stop 
            will be plotted. The default is True.
            
        return_metrics : Boolean, optional
            Whether the metrics computed using the given trailing stop option 
            should be returned or not. If this is true the functions returns
            dfPRR - the dataframe of the strategies returns using the adjusted
            prices and metrics - a tuple of metrics. Otherwise only dfPRR is
            returned. The default is False.
            
        trailing_stop : Boolean, optional
            Whether to use a trailing stop or not in computing the financial
            metrics. The default is False.
            
        plotting : Boolean, optional
            Boolean that determines whether the equity curve is plotted. If 
            show_trailing is True and trailing_stop is False an equity curve
            will still be plotted to show the difference in using a trailing
            stop vs otherwise. The default is False.
            
        print_metrics : Boolean, optional
            If True, prints the metrics of the selected trailing stop option.
            Note that is show_trailing is True trailing stop metrics will still
            be printed out. The default is False.

        Returns
        -------
        dfPRR: pandas dataframe of returns according to the adjusted prices,
            supplemented with equity curve.
        
        metrics: Optional, only returned if return_metrics is True. 
            Tuple containing:
            trading days based total return
            total return
            trading days based CAGR
            CAGR
            Sharpe Ratio
            Volatility

        """
        
        
        assert len(dfP_list) == len(dfAP_list), "Each set of ETF prices must have a set of closed prices."
        
        dfPRR_list = []
        for portfolio_index in range(len(dfP_list)):
            dfPRR = self.backtest(dfP_list[portfolio_index],
                                            dfAP_list[portfolio_index],
                                            trailing_stop = trailing_stop,
                                            show_trailing=False,
                                            laddering = laddering,
                                            plotting = False)
            dfPRR_list.append(dfPRR)
            
            dfPRR.to_csv(r'Results\dfPRR_set_' + str(portfolio_index)+'.csv', header = True, index=True, encoding='utf-8')

        dfPRR_combined = pd.DataFrame(index = dfPRR_list[0].index)

        dfPRR_combined["ALL_R"] = sum(dfPRR_list[i]["ALL_R"] * portfolio_weights_list[i] for i in range(len(dfPRR_list)) )
        dfPRR_combined["DETREND_ALL_R"] = sum(dfPRR_list[i]["DETREND_ALL_R"] * portfolio_weights_list[i] for i in range(len(dfPRR_list)) )
        dfPRR_combined["I"] = sum(dfPRR_list[i]["I"] * portfolio_weights_list[i] for i in range(len(dfPRR_list)) )
        
        print("Combined Portfolio Results")
        
        
        
        if (show_trailing == True) and (trailing_stop == False):
            #Compute and plot trailing stop graph, but do not return metrics
            self.compute_financial_metric_results(dfPRR_combined, trailing_stop= True, plotting = True, print_metrics = True)
        
        metrics = self.compute_financial_metric_results( dfPRR_combined, trailing_stop=trailing_stop, return_metrics = return_metrics, plotting = plotting, print_metrics = print_metrics)
        
        if return_metrics == False:
            return dfPRR_combined
        else:
            return dfPRR_combined, metrics
        
    def compute_financial_metric_results(self, dfPRR, trailing_stop = False, plotting = False, return_metrics = False, print_metrics = False):
        """
        Computes total return, CAGR, Sharpe ratio and volatility of the input
        dataframe of returns

        Parameters
        ----------
        dfPRR : Dataframe of returns, must contain a column 'ALL_R' - 
            containing the sum of all returns of the ETFs, 'DETREND_ALL_R' - 
            containing the detrended versions of 'ALL_R' and 'I' containing the
            equity curve based of the number of units of each ETF selected.
            
        trailing_stop : Boolean, optional
            Whether to use a trailing stop in calculating metrics. The default 
            is False.
            
        plotting : Boolean, optional
            Whether to plot the results. The default is False.
            
        return_metrics : Boolean, optional
            Whether to return the metrics. Returns None if False. The default 
            is False.
            
        print_metrics : Boolean, optional
            Whether to print metrics to the screen. The default is False.

        Returns
        -------
        metrics: Optional, only returned if return_metrics is True. 
            Tuple containing:
            trading days based total return
            total return
            trading days based CAGR
            CAGR
            Sharpe Ratio
            Volatility.

        """
        if trailing_stop == False:
            try:
                sharpe = ((dfPRR['ALL_R'].mean() / dfPRR['ALL_R'].std()) * math.sqrt(252)) 
            except ZeroDivisionError:
                sharpe = 0.0
    
    
            volatility =  ((dfPRR['ALL_R'].std()) * math.sqrt(252)) 
    
            if plotting == True:
                style.use('fivethirtyeight')
                dfPRR.plot(y=['I'], figsize=(10,5), grid=True)
                plt.legend()
                plt.show()
                #plt.savefig(r'Results\%s.png' %(title))
                #plt.close()
    
    
            start = 1
            start_val = start
            end_val = dfPRR['I'].iat[-1]
                
            start_date = self.get_date(dfPRR.iloc[0].name)
            end_date = self.get_date(dfPRR.iloc[-1].name)
            days = (end_date - start_date).days
    
            TotaAnnReturn = (end_val-start_val)/start_val/(days/360)
            TotaAnnReturn_trading = (end_val-start_val)/start_val/(days/252)
                
            CAGR_trading = round(((float(end_val) / float(start_val)) ** (1/(days/252.0))).real - 1,4) #when raised to an exponent I am getting a complex number, I need only the real part
            CAGR = round(((float(end_val) / float(start_val)) ** (1/(days/350.0))).real - 1,4) #when raised to an exponent I am getting a complex number, I need only the real part
    
            if print_metrics == True:
                print("Financial metrics of I:")
                print ("TotaAnnReturn = %f" %(TotaAnnReturn*100))
                print ("CAGR = %f" %(CAGR*100))
                print ("Sharpe Ratio = %f" %(round(sharpe,2)))
                print ("Volatility = %f" %(round(volatility,2)))
            
    
                #Detrending Prices and Returns
                WhiteRealityCheckFor1.bootstrap(dfPRR['DETREND_ALL_R'])
                print ("\n")
                
            
        else:
            
            #If we use a trailing stop
            
            #dfP['Stance'] = 1
            shift = 2 #prices usually overshoot and correct the next day
            
            window1 = int(self.lookback) #short window
            window2 = 3*window1+((3*window1)//20)*2  #long window
            dfPRR['Stance'] = 1
            dfPRR['IncVol'] = 0
            dfPRR['Regime'] = 0
            dfPRR = dfPRR.assign(PercentMvt =(dfPRR['ALL_R']-dfPRR['ALL_R'].shift(shift))/dfPRR['ALL_R'].shift(shift))

            dfPRR.replace({'PercentMvt':{np.nan:0, np.inf:0, -np.inf:0}}, inplace=True) #alternative (2) 
            dfPRR["STD"] = dfPRR.PercentMvt.rolling(window1).std()
            dfPRR["2STD"] = dfPRR.PercentMvt.rolling(window2).std()
            dfPRR.replace({'STD':{np.nan:0, np.inf:0, -np.inf:0}}, inplace=True)
            dfPRR.replace({'2STD':{np.nan:0, np.inf:0, -np.inf:0}}, inplace=True)
            dfPRR['IncVol'] = np.where(dfPRR["STD"] > dfPRR["2STD"], 1, 0) 
            dfPRR = dfPRR.assign(I_MA =  dfPRR['I'].rolling(window2*3).mean()) #close to the classic 200MA
            dfPRR['Regime'] = np.where((dfPRR['I_MA'] > dfPRR['I']), 1, 0)

            #alternatives (1) and (2) are both reasonable
            #(1) and (2) require a regime which can be either ("IncVol") or ("Regime") to impose the percent loss filter
            #(1) measures an unusual returns move with a multiple of the volatility of the returns, (2) uses a multiple of the returns
            #dfPRR['Stance'] = np.where(((dfPRR['PercentMvt'] < -1*np.abs(dfPRR['PercentMvt'].rolling(window2).mean()+.3*dfPRR['STD'])) & (dfPRR['IncVol']>0)), 0, 1) #alternative (1)
            dfPRR['Stance'] = np.where(((dfPRR['PercentMvt'] < -1*np.abs(dfPRR['PercentMvt'].rolling(window2).mean()*1.8)) & (dfPRR['Regime']>0)), 0, 1) #alternative (2), can exchange 'Regime' for 'IncVol'
            dfPRR['ALL_R_filtered'] = dfPRR['ALL_R']*dfPRR['Stance'].shift(1)
            dfPRR = dfPRR.assign(I_filtered =np.cumprod(1+dfPRR['ALL_R_filtered'])) 



            #calculate the financial metrics
            try:
                sharpe = ((dfPRR['ALL_R_filtered'].mean() / dfPRR['ALL_R_filtered'].std()) * math.sqrt(252)) 
            except ZeroDivisionError:
                sharpe = 0.0

            volatility =  ((dfPRR['ALL_R_filtered'].std()) * math.sqrt(252)) 

            if plotting == True:
                style.use('fivethirtyeight')
                dfPRR.plot(y=['I', 'I_filtered'], figsize=(10,5), grid=True)
                plt.legend()
                plt.show()
                #plt.close()

            start = 1
            start_val = start
            end_val = dfPRR['I_filtered'].iat[-1]
                
            start_date = self.get_date(dfPRR.iloc[0].name)
            end_date = self.get_date(dfPRR.iloc[-1].name)
            days = (end_date - start_date).days

            TotaAnnReturn = (end_val-start_val)/start_val/(days/360)
            TotaAnnReturn_trading = (end_val-start_val)/start_val/(days/252)
                
            CAGR_trading = round(((float(end_val) / float(start_val)) ** (1/(days/252.0))).real - 1,4) #when raised to an exponent I am getting a complex number, I need only the real part
            CAGR = round(((float(end_val) / float(start_val)) ** (1/(days/350.0))).real - 1,4) #when raised to an exponent I am getting a complex number, I need only the real part

            if print_metrics == True:
                print("Financial metrics of I:")
                print ("TotaAnnReturn = %f" %(TotaAnnReturn*100))
                print ("CAGR = %f" %(CAGR*100))
                print ("Sharpe Ratio = %f" %(round(sharpe,2)))
                print ("Volatility = %f" %(round(volatility,2)))
            
            
                #Detrending Prices and Returns
                WhiteRealityCheckFor1.bootstrap(dfPRR['DETREND_ALL_R'])
                print ("\n")
        
        if return_metrics == False:
            return
        else:
            TAR_trading_per   = TotaAnnReturn_trading * 100
            TAR_per = TotaAnnReturn*100
            CAGR_trading_per   = CAGR_trading * 100
            CAGR_per = CAGR*100
            sharpe_val = round(sharpe,2)
            vol_val = round(volatility,2)
            
            metrics = (TAR_trading_per, TAR_per, CAGR_trading_per, CAGR_per, sharpe_val, vol_val)
                
            return metrics
