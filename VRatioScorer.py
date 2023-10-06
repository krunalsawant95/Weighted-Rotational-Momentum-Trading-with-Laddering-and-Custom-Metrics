# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 10:33:41 2023

@author: krunal
"""
import numpy as np
import pandas as pd




def compute_vratio_score(dfP,lag=2,window_size=200,cor = 'hom'):
    """
    Takes in a dataframe of prices and computes a score for each row in the 
    data frame. The score combinines the v-ratio the p value of the v-ratio to 
    reflect the magnitude and the statistical significance of the trend. 

    Parameters
    ----------
    dfP : dataframe of prices

    lag : int, optional
        Integer period used to calculate the average differences for the
        variance comparison. The default is 2.
    window_size : int, optional
        The lookback window length used to calculate the v-ratio statistic. For
        statistical significance it must be at least 100 times the lag. The 
        default is 200.
    cor : string, optional
        Calculation type for the correlation. Can be 'hom' or 'het'. The 
        default is 'hom'.

    Returns
    -------
    vratio_score_df: dataframe of combined v-ratio scores with the same shape,
    index and columns as dfP.

    """
    
    def normcdf(X):
        """
        Returns the cumulative normal distribution probability value for X.
        """
        (a1,a2,a3,a4,a5) = (0.31938153, -0.356563782, 1.781477937, -1.821255978, 1.330274429)
        L = abs(X)
        K = 1.0 / (1.0 + 0.2316419 * L)
        w = 1.0 - 1.0 / np.sqrt(2*np.pi)*np.exp(-L*L/2.) * (a1*K + a2*K*K + a3*pow(K,3) + a4*pow(K,4) + a5*pow(K,5))
        if X < 0:
            w = 1.0-w
        return w
    
    def vratio(a, lag = 2, cor = 'hom'):
        """ vratio implementation found in the blog Leinenbock  
        http://www.leinenbock.com/variance-ratio-test/
        """
        #t = (std((a[lag:]) - (a[1:-lag+1])))**2;
        #b = (std((a[2:]) - (a[1:-1]) ))**2;
     
        n = len(a)
        mu  = sum(a[1:n]-a[:n-1])/n;
        m=(n-lag+1)*(1-lag/n);
        #print( mu, m, lag)
        b=sum(np.square(a[1:n]-a[:n-1]-mu))/(n-1)
        t=sum(np.square(a[lag:n]-a[:n-lag]-lag*mu))/m
        vratio = t/(lag*b);
     
        la = float(lag)
         
        if cor == 'hom':
            varvrt=2*(2*la-1)*(la-1)/(3*la*n)
            
        elif cor == 'het':
            varvrt=0;
            sum2=sum(np.square(a[1:n]-a[:n-1]-mu));
            for j in range(lag-1):
                sum1a=np.square(a[j+1:n]-a[j:n-1]-mu);
                sum1b=np.square(a[1:n-j]-a[0:n-j-1]-mu)
                sum1=np.dot(sum1a,sum1b);
                delta=sum1/(sum2**2);
                varvrt=varvrt+((2*(la-j)/la)**2)*delta
     
        zscore = (vratio - 1) / np.sqrt(float(varvrt))
        pval = normcdf(zscore);
     
        return  vratio, zscore, pval            
    
    #Setup dataframe
    vratio_score_df = pd.DataFrame(index = dfP.index, columns = dfP.columns)    
    
    for col in dfP.columns:
        #vratio code is for ndarrays. Differencing with an dataframe index
        #will result in nan values. Need to modify v-ratio or just extract
        #ndarray with .values
        prices = dfP[col].values
        n = len(prices)
        
        #Compute score and significance for a rolling window of window_size
        for i in range(window_size,n):
            window_price = prices[i-window_size:i]
            vratio_val, z_score, pval = vratio(window_price,lag,cor)
            
            
            #Combining scores
            if vratio_val < 1:
                #Mean reverting or random walk behaviour
                #Significant for pval results <= 0.05
                #Penalize by multiplying by pval-1 to make negavtive
                #(Stronger the trend, the bigger the negative result)
                #Penalize the further below 1 by multiplying by (1 - val)
                #(The faster the reversion, the bigger the negative result)
                vratio_score = (1 - vratio_val) * (pval -1)
            else:
                #Momentum/trending behaviour
                #Significant normcdf pval >= 0.95 with this formulation
                #Evaluate and reward by multiplying by normcdf pval
                vratio_score = vratio_val*pval
            vratio_score_df.loc[dfP.index[i],col] = vratio_score
    
    return vratio_score_df

