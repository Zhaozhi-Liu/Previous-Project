import numpy as np
from scipy.stats import norm

class Euro_call:
    def __init__(self,spot,strike_price,r,sigma,T):
        self.spot = spot
        self.strike_price = strike_price
        self.r = r
        self.sigma = sigma
        self.T = T
        self.rng = np.random.default_rng()
        self.df = np.exp(-self.r * self.T)
    @property       
    def price(self):
        # BS price
        d_positive = (np.log(self.spot / self.strike_price) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d_negative = d_positive - self.sigma * np.sqrt(self.T)
        return self.spot * norm.cdf(d_positive) - np.exp(-self.r * self.T) * norm.cdf(d_negative) * self.strike_price
    
    @property
    def delta(self):
        # BS delta
        d_positive = (np.log(self.spot / self.strike_price) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        return norm.cdf(d_positive)
    
    @property
    def gamma(self):
        # BS gamma
        d_positive = (np.log(self.spot/self.strike_price) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        return norm.pdf(d_positive) / (self.spot * self.sigma * self.T ** 0.5)
    
    @property
    def vega(self):
        # BS vega
        d_positive = (np.log(self.spot / self.strike_price) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        return self.spot * (self.T**0.5) * norm.pdf(d_positive)
    
    def stock_price(self, N):
        sample = self.rng.normal(size = N)
        return self.spot * np.exp((self.r - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(self.T) * sample)
    
    def _stock_price(self, N):
        # return wiener process at time T and stock price 
        sample = self.rng.normal(size = N)
        return np.sqrt(self.T) * sample,self.spot * np.exp((self.r - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(self.T) * sample)
    
    def Monte_Carlo_price(self, N, antithetic_variates = False, control_variates = False):
        if antithetic_variates:
            sample = self.rng.normal(size = N)
            price_sample_1 = self.spot * np.exp((self.r - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(self.T)*sample)
            payoff_1 = np.maximum(price_sample_1-self.strike_price,0)
            price_sample_2 = self.spot * np.exp((self.r - 0.5 * self.sigma**2) * self.T - self.sigma * np.sqrt(self.T)*sample)
            payoff_2 = np.maximum(price_sample_2-self.strike_price,0)
            p = 0.5 * (np.mean(payoff_1) + np.mean(payoff_2)) * np.exp(-self.r * self.T)
            return p
        
        if control_variates:
            sample = self.rng.normal(size = N)
            price_sample = self.spot * np.exp((self.r - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(self.T) * sample) 
            mean_1 = np.mean(price_sample) * self.df # mean of discounted sample price
            payoff = np.maximum(price_sample-self.strike_price,0) 
            mean_2 = np.mean(payoff) * self.df     # mean of discounted sample payoff
            cov = np.sum((price_sample * self.df - mean_1) * (payoff * self.df - mean_2)) / np.sum((price_sample * self.df - mean_1)**2)
            p = mean_2 - cov * (mean_1 - self.spot) 
            return p
        
        if antithetic_variates and control_variates:
            sample = self.rng.normal(size = N)
            p = []
            for i in [-1,1]:
                price_sample = self.spot * np.exp((self.r - 0.5 * self.sigma**2) * self.T + i * self.sigma * np.sqrt(self.T) * sample) 
                mean_1 = np.mean(price_sample) * self.df
                payoff = np.maximum(price_sample-self.strike_price,0) 
                mean_2 = np.mean(payoff) * self.df
                cov = np.sum((price_sample * self.df - mean_1) * (payoff * self.df - mean_2)) / np.sum((price_sample * self.df - mean_1)**2)
                p.append(mean_2 - cov * (mean_1 - self.spot))   
            return np.mean(p)
        
        payoff = np.maximum(self.stock_price(N) - self.strike_price,0)
        p = np.mean(payoff) * np.exp(-self.r * self.T)
        return p
        
    def pathwise_delta(self, N):
        s_T = self.stock_price(N)
        return self.df * np.mean((s_T / self.spot) * (s_T > self.strike_price))
    
    def pathwise_vega(self, N):
        w_T,s_T = self._stock_price(N)
        return self.df * np.mean((-self.sigma * self.T + w_T) * s_T * (s_T > self.strike_price))
    
    def pw_lr_gamma(self, N):
        w_T,s_T = self._stock_price(N)
        return self.df * np.mean((s_T / self.spot**2) * (w_T / (self.sigma * self.T) - 1) * (s_T > self.strike_price))
        
    def lr_pw_gamma(self, N):
        w_T,s_T = self._stock_price(N)
        return self.df * np.mean((self.strike_price * w_T / (self.spot**2 * self.sigma * self.T)) * (s_T > self.strike_price))
    
    def likelihood_delta(self, N):
        w_T,s_T = self._stock_price(N)
        payoff = np.maximum(s_T - self.strike_price,0)
        return self.df * np.mean(payoff * (w_T/(self.spot * self.sigma * self.T)))
    
    def likelihood_gamma(self, N):
        w_T,s_T = self._stock_price(N)
        payoff = np.maximum(s_T - self.strike_price,0)
        d = self.df * 1/((self.spot**2) * self.sigma * self.T)
        return d * np.mean(payoff * (w_T**2/(self.sigma * self.T)-w_T-1/self.sigma))
    
    def likelihood_vega(self, N):
        w_T,s_T = self._stock_price(N)
        payoff = np.maximum(s_T - self.strike_price, 0)
        return self.df * np.mean(payoff * (w_T**2 / (self.sigma * self.T) - w_T - 1 / self.sigma))
    
    def Monte_Carlo_delta(self, N, dx = 3.5):
        self.spot += dx
        p1 = self.Monte_Carlo_price(N, control_variates = True)
        self.spot -= 2 * dx
        p2 = self.Monte_Carlo_price(N, control_variates = True)
        self.spot += dx
        return (p1 - p2) / (2 * dx)
    
    def Monte_Carlo_gamma(self, N, dx = 3.5):
        self.spot += dx
        p1 = self.Monte_Carlo_price(N, control_variates = True)
        self.spot -= 2 * dx
        p2 = self.Monte_Carlo_price(N, control_variates = True)
        self.spot += dx
        p3 = self.Monte_Carlo_price(N, control_variates = True)
        return (p1 + p2 - 2 * p3) / dx**2
    
    def Monte_Carlo_vega(self,N,dx = 0.01):
        self.sigma += dx
        p1 = self.Monte_Carlo_price(N, control_variates = True)
        self.sigma -= 2 * dx
        p2 = self.Monte_Carlo_price(N, control_variates = True)
        self.sigma += dx
        return (p1 - p2) / (2 * dx)