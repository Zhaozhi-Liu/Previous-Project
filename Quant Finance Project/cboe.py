import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime,date,timedelta
from functions import *
mpl.rcParams["figure.dpi"] = 480
mpl.rcParams["figure.figsize"] = (10,6)
plt.style.use("ggplot")
spx = yf.Ticker("^SPX")
time_chain = list(map(lambda x:datetime.strptime(x, "%Y-%m-%d"),spx.options))

class CBOE:

    def __init__(self,maturity):
        self.today = datetime.strptime(datetime.now().strftime("%Y-%m-%d"), "%Y-%m-%d")
        self.spx = yf.Ticker("^SPX") 
        self.r = yf.Ticker("^IRX").history(period="5d")["Close"].iloc[-1]*0.01
        self.days = abs((maturity - self.today).days)
        self.T = self.days/365
        self.options = spx.option_chain(maturity.strftime("%Y-%m-%d"))
        self.calls = self.options.calls[["contractSymbol","strike","lastPrice","bid","ask","volume"]]
        self.puts = self.options.puts[["contractSymbol","strike","lastPrice","bid","ask","volume"]]
        self.k0,self.call_price,self.put_price = self._k0()
        self.variance = self.vol() 
        
    def _k0(self):
        call_df = self.options.calls.loc[:,["strike","lastPrice"]]
        put_df = self.options.puts.loc[:,["strike","lastPrice"]]
        merged = pd.merge(call_df,put_df,how = "inner",on="strike",suffixes = ["_call","_put"])
        merged["diff"] = abs(merged["lastPrice_call"] - merged["lastPrice_put"]) 
        merged = merged.iloc[::-1].reset_index()
        k0,call_price,put_price = merged.iloc[merged["diff"].idxmin(),:][["strike","lastPrice_call","lastPrice_put"]].values
        return k0,call_price,put_price
    
    @property
    def F(self):
        return self.k0 + np.exp(self.r*self.T)*(self.call_price-self.put_price)   
        
    
        
    def vol(self):
        cond1 = self.puts["strike"] <= self.k0
        cond2 = self.puts["ask"] > 0
        cond3 = self.puts["bid"] > 0
        cond4 = self.puts["volume"] > 0
        puts_OTM_CBOE = self.puts[cond1 & cond2 & cond3 & cond4]
        diff_k = puts_OTM_CBOE["strike"].diff().values
        T_1 = np.sum((puts_OTM_CBOE["bid"].values[:-1] + puts_OTM_CBOE["ask"].values[:-1])/2 * np.exp(self.r*self.T)/(puts_OTM_CBOE["strike"].values[:-1]**2) * diff_k[1:])
        T_1 -= (puts_OTM_CBOE["bid"].values[-2] + puts_OTM_CBOE["ask"].values[-2])/2 * np.exp(self.r*self.T)/(puts_OTM_CBOE["strike"].values[-2]**2)
        
        cond1 = self.calls["strike"] >= self.k0
        cond2 = self.calls["ask"] > 0
        cond3 = self.calls["bid"] > 0
        cond4 = self.calls["volume"] > 0
        calls_OTM_CBOE = self.calls[cond1 & cond2 & cond3 & cond4]
        
        diff_k = calls_OTM_CBOE["strike"].diff().values
        T_3 = np.sum((calls_OTM_CBOE["bid"].values[1:] + calls_OTM_CBOE["ask"].values[1:])/2 * np.exp(self.r*self.T)/(calls_OTM_CBOE["strike"].values[1:]**2) * diff_k[1:])
        T_3 -= (calls_OTM_CBOE["bid"].values[1] + calls_OTM_CBOE["ask"].values[1])/2 * np.exp(self.r*self.T)/(calls_OTM_CBOE["strike"].values[1]**2)

        T_2 = (puts_OTM_CBOE["lastPrice"].values[-1] + calls_OTM_CBOE["lastPrice"].values[0])* np.exp(self.r*self.T) / (self.k0**2) * 2
        T_4 = (self.F/self.k0-1)**2
        
        vix = (2*T_1 + T_2 + 2*T_3 - T_4)/self.T
        return vix
        
        
        
        
        
        