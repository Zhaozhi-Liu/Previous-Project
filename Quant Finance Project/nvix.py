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

class NVIX:

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
        self.table,self.points = self.construct_dataframe()
        self.variance = self.vol() 
        
    def _k0(self):
        call_df = self.options.calls.loc[:,["strike","lastPrice"]]
        put_df = self.options.puts.loc[:,["strike","lastPrice"]]
        # if the merged dataframe is empty, please change an earlier or later maturity
        merged = pd.merge(call_df,put_df,how = "inner",on="strike",suffixes = ["_call","_put"])
        merged["diff"] = abs(merged["lastPrice_call"] - merged["lastPrice_put"]) 
        merged = merged.iloc[::-1].reset_index()
        k0,call_price,put_price = merged.iloc[merged["diff"].idxmin(),:][["strike","lastPrice_call","lastPrice_put"]].values
        return k0,call_price,put_price
    
    @property
    def F(self):
        return self.k0 + np.exp(self.r*self.T)*(self.call_price-self.put_price)   
    
    def d2(self,k,sigma):
        return -k/(sigma*np.sqrt(self.T)) - (sigma*np.sqrt(self.T))/2

    def P_BS(self,k,sigma):
        d2_ = self.d2(k,sigma)
        d1 = d2_ + (sigma*np.sqrt(self.T))
        return self.F * np.exp(k) * norm.cdf(-d2_) - self.F * norm.cdf(-d1)

    def P_implied_vol(self, k,price):
        obj_f = lambda sigma: self.P_BS(k,sigma) - price * np.exp(self.r*self.T)
        root = brentq(obj_f,0.01,1)
        return root

    def C_BS(self,k,sigma):
        d2_ = self.d2(k,sigma)
        d1 = d2_ + (sigma*np.sqrt(self.T))
        return -self.F * np.exp(k) * norm.cdf(d2_) + self.F * norm.cdf(d1)

    def C_implied_vol(self, k,price):
        obj_f = lambda sigma: self.C_BS(k,sigma) - price * np.exp(self.r*self.T)
        root = brentq(obj_f,0.01,1)
        return root    
    
    def construct_dataframe(self):
        cond1 = self.calls["strike"] > self.k0
        cond2 = self.calls["ask"] > 0
        cond3 = self.calls["bid"] > 0
        cond4 = self.calls["ask"] / self.calls["bid"] < 2 
        cond5 = self.calls["volume"] > 0
        calls_OTM = self.calls[cond1 & cond2 & cond3 & cond4 & cond5]
        
        cond1 = self.puts["strike"] <= self.k0
        cond2 = self.puts["ask"] > 0
        cond3 = self.puts["bid"] > 0
        cond4 = self.puts["ask"] / self.puts["bid"] < 2 
        cond5 = self.puts["volume"] > 0
        puts_OTM = self.puts[cond1 & cond2 & cond3 & cond4 & cond5]
    
        copy = calls_OTM.copy()
#         copy["imp_vol"] = copy.apply(lambda x:self.C_implied_vol(np.log(x["strike"]/self.F),x["lastPrice"]),axis = 1)**2
        copy["imp_vol"] = copy.apply(lambda x:self.C_implied_vol(np.log(x["strike"]/self.F),(x["bid"]+x["ask"])/2),axis = 1)**2
        copy["d2"] = copy.apply(lambda x: self.d2(np.log(x["strike"]/self.F),x["imp_vol"]**0.5),axis = 1)
        calls_OTM = copy
    
        copy = puts_OTM.copy()
#         copy["imp_vol"] = copy.apply(lambda x:self.P_implied_vol(np.log(x["strike"]/self.F),x["lastPrice"]),axis = 1)**2
        copy["imp_vol"] = copy.apply(lambda x:self.P_implied_vol(np.log(x["strike"]/self.F),(x["bid"]+x["ask"])/2),axis = 1)**2
        copy["d2"] = copy.apply(lambda x: self.d2(np.log(x["strike"]/self.F),x["imp_vol"]**0.5),axis = 1)
        puts_OTM = copy
        
        i,j = longest_dinterval(calls_OTM)
        calls_OTM = calls_OTM.iloc[i:j+1,:]
        i,j = longest_iinterval(puts_OTM)
        puts_OTM = puts_OTM.iloc[i:j+1,:]    
        
        df_combine = pd.concat([puts_OTM,calls_OTM],axis = 0)
        df_combine = df_combine.sort_values(by = "d2")
        
        x = df_combine["d2"]
        y = df_combine["imp_vol"]
        points = np.column_stack((x, y))
        diff = np.diff(points,axis = 0,prepend = np.nan)
        diff_x = diff[:,0]
        diff_y = diff[:,1]
        
        l_plus = np.sqrt((diff_x[2:]**2 + diff_y[2:]**2))
        l = np.sqrt((diff_x[1:-1]**2 + diff_y[1:-1]**2))
        numerator = diff_x[2:]/l_plus - diff_x[1:-1]/l
        denominator = diff_y[2:]/l_plus - diff_y[1:-1]/l
        b = -numerator/denominator
        b = np.concatenate(([0],b,[0]))
        df_combine["b"] = b        
        
        c = (3*diff_y[1:] - diff_x[1:]*b[1:] - 2*diff_x[1:]*b[:-1])/diff_x[1:]**2
        c = np.concatenate((c,[0]))
        df_combine["c"] = c    
        
        d = (diff_y[1:] - b[:-1]*diff_x[1:] - c[:-1]*diff_x[1:]**2)/diff_x[1:]**3
        d = np.concatenate((d,[0]))
        df_combine["d"] = d
        return df_combine,points
    
    def _y(self,x):
        df_combine = self.table
        if x<=df_combine["d2"].iloc[0]:
            return df_combine["imp_vol"].iloc[0]
        elif x>=df_combine["d2"].iloc[-1]:
            return df_combine["imp_vol"].iloc[-1]
        array = df_combine["d2"].values
        ind = np.searchsorted(array,x,side = "right")-1
        x_j = array[ind]
        a = df_combine["imp_vol"].values[ind]
        b = df_combine["b"].values[ind]
        c = df_combine["c"].values[ind]
        d = df_combine["d"].values[ind]
        return a+b*(x-x_j)+c*(x-x_j)**2+d*(x-x_j)**3
    
    def graph(self):
        tmp_list = np.linspace(self.points[0,0]-1,self.points[-1,0]+1,10000)
        plt.plot(tmp_list,[self._y(x) for x in tmp_list],"b-")
        plt.scatter(self.points[:,0],self.points[:,1])
        plt.xlabel("$d_2$")
        plt.ylabel("$\sigma^2$")
        
    def vol(self):
        df_combine = self.table
        x = df_combine["d2"].values
        phi_plus = norm.pdf(x[1:])
        phi = norm.pdf(x[:-1])
        A = norm.cdf(x[1:]) - norm.cdf(x[:-1])
        B = -(phi_plus - phi) - x[:-1]*A
        C = (
            -(x[1:]*phi_plus-x[:-1]*phi) + 2*x[:-1]*(phi_plus-phi) + 
             (1+x[:-1]**2)*A
        )
        D = (
            (1-x[1:]**2)*phi_plus - (1-x[:-1]**2)*phi + 
            3*x[:-1]*(x[1:]*phi_plus - x[:-1]*phi) - 
            3*(1+x[:-1]**2)*(phi_plus-phi) - 
            x[:-1]*(3+x[:-1]**2)*A
        )
        a = df_combine["imp_vol"].values
        b = df_combine["b"].values
        c = df_combine["c"].values
        d = df_combine["d"].values
        part1 = np.sum(a[:-1] * A + b[:-1] * B + c[:-1] * C + d[:-1] * D)
        part2 = a[0] * norm.cdf(x[0]) + a[-1]*(1-norm.cdf(x[-1]))
        return part1 + part2   