import numpy as np
from numba import jit
from scipy.stats import norm

@jit
def _stock_price(self_T, self_r, self_sigma, self_spot, N, M, sample):
    dt = self_T/M
    S = np.zeros_like(sample)
    S[:,0] = self_spot
    for i in range(N):
        for j in range(1,M):
            S[i,j] = S[i,j-1] * np.exp((self_r-0.5 * self_sigma**2) * dt + self_sigma * np.sqrt(dt) * sample[i,j-1])
    return S

@jit
def _processed_sample(self_b, self_type, stock_sample):
    # this function will return an array with element equal to 1 or 0
    # the output array has the same length of the number of path of stock price
    # if the i-th element of the output array is 0, then the option is over on the i-th path
    # if the i-th element of the output array is 1, then the payoff will be given at maturity on the i-th path
    if self_type == "down-and-out":
        N = len(stock_sample)
        tmp = np.zeros(N)
        for i in range(N):
            tmp[i] = np.min(stock_sample[i]) > self_b

    elif self_type == "down-and-in":
        N = len(stock_sample)
        tmp = np.zeros(N)
        for i in range(N):
            tmp[i] = np.min(stock_sample[i]) <= self_b

    elif self_type == "up-and-out":
        N = len(stock_sample)
        tmp = np.zeros(N)
        for i in range(N):
            tmp[i] = np.max(stock_sample[i]) < self_b

    elif self_type == "up-and-in":
        N = len(stock_sample)
        tmp = np.zeros(N)
        for i in range(N):
            tmp[i] = np.max(stock_sample[i]) >= self_b
    return tmp
    
@jit
def _efficient_mc_sample(self_T, self_sigma, N, M, barrier, stock_sample, uniform_sample, self_type):
    # this function will return an array with element equal to 1 or 0
    # the output array has the same length of the number of path of stock price
    # if the i-th element of the output array is 0, then the option is over on the i-th path
    # if the i-th element of the output array is 1, then the payoff will be given at maturity on the i-th path
    dt = self_T/M
    P = np.zeros_like(stock_sample)
    S = stock_sample
    u = uniform_sample
    if self_type == "up-and-out":
        tmp = np.ones(N)
        for j in range(N):
            for i in range(1,M):
                P[j,i] = np.exp(-2 * ((barrier - S[j,i-1]) * (barrier - S[j,i])) / ((self_sigma * S[j,i-1])**2 * dt))
                if S[j,i] >= barrier or P[j,i] >= u[j,i]:
                    tmp[j] = 0
                    break
    elif self_type == "up-and-in":
        tmp = np.zeros(N)
        for j in range(N):
            for i in range(1,M):
                P[j,i] = np.exp(-2 * ((barrier - S[j,i-1]) * (barrier - S[j,i])) / ((self_sigma * S[j,i-1])**2 * dt))
                if S[j,i] >= barrier or P[j,i] >= u[j,i]:
                    tmp[j] = 1
                    break
                    
    elif self_type == "down-and-out":
        tmp = np.ones(N)
        for j in range(N):
            for i in range(1,M):
                P[j,i] = np.exp(-2 * ((barrier - S[j,i-1]) * (barrier - S[j,i])) / ((self_sigma * S[j,i-1])**2 * dt))
                if S[j,i] <= barrier or P[j,i] >= u[j,i]:
                    tmp[j] = 0
                    break
                    
    elif self_type == "down-and-in":
        tmp = np.zeros(N)
        for j in range(N):
            for i in range(1,M):
                P[j,i] = np.exp(-2 * ((barrier - S[j,i-1]) * (barrier - S[j,i])) / ((self_sigma * S[j,i-1])**2 * dt))
                if S[j,i] <= barrier or P[j,i] >= u[j,i]:
                    tmp[j] = 1
                    break
    return tmp

@jit
def _CE_mc_sample(self_T, self_sigma, N, M, barrier, stock_sample, self_type):
    dt = self_T/M
    S = stock_sample 
    prob = np.ones(N)
    if self_type == "up-and-out":
        for j in range(N):
            for i in range(1,M):
                P = 1 - np.exp(-2 * ((barrier - S[j,i-1]) * (barrier - S[j,i])) / ((self_sigma * S[j,i-1])**2 * dt))
                prob[j] *= P if (S[j,i-1]<barrier and S[j,i]<barrier) else 0
        return prob
    elif self_type == "down-and-out":
        for j in range(N):
            for i in range(1,M):
                P = 1 - np.exp(-2 * ((barrier - S[j,i-1]) * (barrier - S[j,i])) / ((self_sigma * S[j,i-1])**2 * dt))
                prob[j] *= P if (S[j,i-1]>barrier and S[j,i]>barrier) else 0
        return prob
    else:
        return None
                
# BS formula for put and call
def BS_put(price,strike,r,T,sigma):
    d_positive = (np.log(price / strike) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d_negative = d_positive - sigma * np.sqrt(T)
    return -price * norm.cdf(-d_positive) + np.exp(-r * T) * norm.cdf(-d_negative) * strike        
def BS_call(price,strike,r,T,sigma):
    d_positive = (np.log(price / strike) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d_negative = d_positive - sigma * np.sqrt(T)
    return price * norm.cdf(d_positive) - np.exp(-r * T) * norm.cdf(d_negative) * strike   

class Barrier_option:
    def __init__(self, spot, strike, r, sigma, T, barrier, opt_type, barrier_type):
        self.spot = spot # initial price
        self.strike = strike # strike price
        self.r = r # intereste rate
        self.sigma = sigma # volatility
        self.T = T # time to expire
        self.rng = np.random.default_rng() # random sample generator
        self.df = np.exp(-self.r * self.T) # discounted factor
        self.b = barrier # barrier of the option
        # payoff function of corresponding vanilla option
        if opt_type == "call":
            self.payoff = lambda path: np.maximum(path - strike, 0)
        else:
            self.payoff = lambda path: np.maximum(strike - path, 0)
        self.type = barrier_type # barrier type
        self.opt_type = opt_type # option type
        self.h = 1e-4 # step size of numerical differentiate
        
    @property
    def European_price(self):
        # BS price
        d_positive = (np.log(self.spot / self.strike) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d_negative = d_positive - self.sigma * np.sqrt(self.T)
        if self.opt_type == "call":
            return self.spot * norm.cdf(d_positive) - np.exp(-self.r * self.T) * norm.cdf(d_negative) * self.strike
        else:
            return -self.spot * norm.cdf(-d_positive) + np.exp(-self.r * self.T) * norm.cdf(-d_negative) * self.strike 
    
    @staticmethod
    def delta_func(sign, s, T, sigma, r):
        # auxiliary function for calculating
        return 1 / (sigma * np.sqrt(T)) * (np.log(s) + (r + sign * 0.5 * sigma**2) * T)

    def _delta(self, sign, s):
        return self.delta_func(sign, s, self.T, self.sigma, self.r)
    
    # Thereotical price
    @property
    def price(self):
        N = norm.cdf
        S = self.spot
        K = self.strike
        si = self.sigma
        r = self.r
        B  = self.b
        T = self.T
        if self.opt_type == "call":
            if self.type == "down-and-out":
                if B <= K:
                    return int(S > B) * BS_call(S, K, r, T, si) - S * int(S>B) * (B/S)**(2 * r / si**2) * BS_call(B/S,K/B,r,T,si)
                else:
                    term1 = S * int(S > B) * N(self._delta(1,S/B))
                    term2 = -self.df * K * int(S > B) * N(self._delta(-1,S/B))
                    term3 = -B * int(S > B) * (B/S)**(2 * r / si**2) * N(self._delta(1,B/S))
                    term4 = self.df * K * int(S > B) * (S/B)**(1-2 * r / si**2) * N(self._delta(-1,B/S))
                    return term1 + term2 + term3 + term4
            elif self.type == "down-and-in":
                if B <= K:
                    self.type = "down-and-out"
                    aa = self.European_price - self.price
                    self.type = "down-and-in"
                    return aa
                else:
                    self.type = "down-and-out"
                    aa = self.European_price - self.price
                    self.type = "down-and-in"
                    return aa                    
            if self.type == "up-and-out":
                if B <= K:
                    return 0
                else:
                    term1 = N(self._delta(1,S/K)) - N(self._delta(1,S/B))
                    term2 = - (B/S)**(1+2 * r / si**2) * (N(self._delta(1,B**2/(K*S))) - N(self._delta(1,B/S)))
                    term3 = N(self._delta(-1,S/K)) - N(self._delta(-1,S/B))
                    term4 = - (S/B)**(1-2 * r / si**2) * (N(self._delta(-1,B**2/(K*S))) - N(self._delta(-1,B/S)))                    
                    return int(S<B) * (S*(term1+term2) - K*self.df*(term3+term4))
            elif self.type == "up-and-in":
                if B <= K:
                    return self.European_price
                else:
                    self.type = "up-and-out"
                    aa = self.European_price - self.price
                    self.type = "up-and-in"
                    return aa
        else:
            if self.type == "down-and-out":
                if B <= K:
                    term1 = BS_put(S, K, r, T, si) + S * N(-self._delta(1,S/B))
                    term2 = -B * (B/S)**(2 * r / si**2) * (N(self._delta(1,B**2/(K*S))) - N(self._delta(1,B/S)))
                    term3 = -self.df * K * N(-self._delta(-1,S/B))
                    term4 = self.df * K * (S/B)**(1-2 * r / si**2) * (N(self._delta(-1,B**2/(K*S))) - N(self._delta(-1,B/S)))
                    return int(S>B) * (term1 + term2 + term3 + term4)
                    
                else:
                    return 0
            elif self.type == "down-and-in":
                if B <= K:
                    self.type = "down-and-out"
                    aa = self.European_price - self.price
                    self.type = "down-and-in"
                    return aa
                else:
                    return self.European_price
            if self.type == "up-and-out":
                if B <= K:
                    term1 = -N(-self._delta(1,S/B)) + (B/S)**(1+2 * r / si**2) * N(-self._delta(1,B/S))
                    term1 = S * term1
                    term2 = -N(-self._delta(-1,S/B)) + (S/B)**(1-2 * r / si**2) * N(-self._delta(-1,B/S))
                    term2 = self.df * K * term2
                    return int(S<B) * (term1 - term2)
                else:
                    term1 = BS_put(S, K, r, T, si)
                    term2 = S * (B/S)**(1+2 * r / si**2) * N(-self._delta(1,B**2/(S*K) ))
                    term3 = -K * self.df * (S/B)**(1-2 * r / si**2) * N(-self._delta(-1,B**2/(S*K) ))
                    return int(S<B) * (term1 + term2 + term3)
            elif self.type == "up-and-in":
                if B <= K:
                    self.type = "up-and-out"
                    aa = self.European_price - self.price
                    self.type = "up-and-in"
                    return aa
                else:
                    self.type = "up-and-out"
                    aa = self.European_price - self.price
                    self.type = "up-and-in"
                    return aa 
                
    # Greeks (approx equal to the thereotical value)
    @property
    def delta(self):
        s_ = self.spot
        self.spot += self.h * 0.5 # price(S_0 + h/2)
        s_plus = self.price
        self.spot -= self.h
        s_minus = self.price      # price(S_0 + h/2)
        self.spot = s_
        return (s_plus - s_minus) / self.h
    
    @property 
    def gamma(self):
        s_ = self.spot
        self.spot += self.h  # price(S_0 + h)
        s_plus = self.price
        self.spot = s_
        self.spot -= self.h
        s_minus = self.price # price(S_0 + h)
        self.spot = s_
        return (s_plus + s_minus - 2 * self.price) / self.h**2    
    
    @property
    def vega(self):
        s_ = self.sigma
        self.sigma += self.h * 0.5 # price(sigma + h/2)
        s_plus = self.price
        self.sigma -= self.h
        s_minus = self.price      # price(sigma + h/2)
        self.sigma = s_
        return (s_plus - s_minus) / self.h        
    
    
    def _sample(self, N, M, dist):
        if dist == "normal":
            return self.rng.normal(size = (N, M))
        if dist == "uniform":
            return self.rng.uniform(size = (N, M))
    
    def stock_path(self,N,M):
        sample = self.rng.normal(size = (N, M))
        return _stock_price(self.T, self.r, self.sigma, self.spot, M, sample)
        
    # Monte Carlo price
    def MC_price(self, N, M):
        sample = self._sample(N, M, "normal")
        S = _stock_price(self.T, self.r, self.sigma, self.spot, N, M, sample)
        tmp =  _processed_sample(self.b, self.type, S)
        return np.mean(self.payoff(S[:,-1]) * tmp) * self.df
    
    def MC_delta(self, N, M):
        dt = self.T/M
        sample = self._sample(N, M, "normal")
        S = _stock_price(self.T, self.r, self.sigma, self.spot, N, M, sample)
        tmp = _processed_sample(self.b, self.type, S)
        return np.mean(self.payoff(S[:,-1]) * tmp * sample[:,0] / (self.spot * self.sigma * dt**0.5)) * self.df
    
    def MC_gamma(self, N, M):
        dt = self.T/M
        sample = self._sample(N, M, "normal")
        S = _stock_price(self.T, self.r, self.sigma, self.spot, N, M, sample)
        tmp = _processed_sample(self.b, self.type, S)    
        score1 = (sample[:,0]**2-1)/(self.spot**2 * self.sigma**2 * dt)
        score2 = sample[:,0] / (self.spot**2 * self.sigma * dt**0.5)
        score = score1 - score2
        return np.mean(self.payoff(S[:,-1]) * tmp * score) * self.df  

    def MC_vega(self, N, M):
        dt = self.T/M
        sample = self._sample(N, M, "normal")
        S = _stock_price(self.T, self.r, self.sigma, self.spot, N, M, sample)
        tmp = _processed_sample(self.b, self.type, S)
        score = (sample[:,:-1]**2-1)/self.sigma -  sample[:,:-1]*dt**0.5
        score = np.sum(score, axis = 1)
        return np.mean(self.payoff(S[:,-1]) * tmp * score) * self.df     
    
    # More efficient Monte Carlo Method
    def EMC_price(self, N, M):
        sample = self._sample(N, M, "normal")
        u = self._sample(N, M, "uniform")
        S = _stock_price(self.T, self.r, self.sigma, self.spot, N, M, sample)
        tmp = _efficient_mc_sample(self.T, self.sigma, N, M, self.b, S, u, self.type)
        return np.mean(self.payoff(S[:,-1]) * tmp) * self.df
    
    def EMC_delta(self, N, M):
        dt = self.T/M
        sample = self._sample(N, M, "normal")
        S = _stock_price(self.T, self.r, self.sigma, self.spot, N, M, sample)
        u = self._sample(N, M, "uniform")
#         k = M//5
        tmp = _efficient_mc_sample(self.T, self.sigma, N, M, self.b, S, u, self.type)
        return np.mean(self.payoff(S[:,-1]) * tmp * sample[:,0] / (self.spot * self.sigma * dt**0.5)) * self.df    
#         return np.mean(self.payoff(S[:,-1]) * tmp * sample[:,k-1] / (self.spot * self.sigma * (dt*k)**0.5)) * self.df

    def EMC_gamma(self, N, M):
        dt = self.T/M
        sample = self._sample(N, M, "normal")
        S = _stock_price(self.T, self.r, self.sigma, self.spot, N, M, sample)
        u = self._sample(N, M, "uniform")
        tmp = _efficient_mc_sample(self.T, self.sigma, N, M, self.b, S, u, self.type)  
#         k = M//5
        score1 = (sample[:,0]**2-1)/(self.spot**2 * self.sigma**2 * dt)
        score2 = sample[:,0] / (self.spot**2 * self.sigma * dt**0.5)
#         score1 = (sample[:,k-1]**2-1)/(self.spot**2 * self.sigma**2 * (dt*k))
#         score2 = sample[:,k-1] / (self.spot**2 * self.sigma * (dt*k)**0.5)
        score = score1 - score2
        return np.mean(self.payoff(S[:,-1]) * tmp * score) * self.df     
    
    def EMC_vega(self, N, M):
        dt = self.T/M
        sample = self._sample(N, M, "normal")
        S = _stock_price(self.T, self.r, self.sigma, self.spot, N, M, sample)
        u = self._sample(N, M, "uniform")
        tmp = _efficient_mc_sample(self.T, self.sigma, N, M, self.b, S, u, self.type)
        score = (sample[:,:-1]**2-1)/self.sigma -  sample[:,:-1]*dt**0.5
        score = np.sum(score, axis = 1)
        return np.mean(self.payoff(S[:,-1]) * tmp * score) * self.df     
    
    
    def CEMC_price(self, N, M):
        dt = self.T/M
        sample = self._sample(N, M, "normal")
        S = _stock_price(self.T, self.r, self.sigma, self.spot, N, M, sample)
        prob = _CE_mc_sample(self.T, self.sigma, N, M, self.b, S, self.type)
        if prob is None:
            raise MyException("Error: Only for knock-out type, use in-out parity to calculate the price")
        return np.mean(self.payoff(S[:,-1]) * prob) * self.df
    
    def CEMC_delta(self, N, M):
        dt = self.T/M
        sample = self._sample(N, M, "normal")
        S = _stock_price(self.T, self.r, self.sigma, self.spot, N, M, sample)
        prob = _CE_mc_sample(self.T, self.sigma, N, M, self.b, S, self.type)
        if prob is None:
            raise MyException("Error: Only for knock-out type, use in-out parity to calculate the price")
#         k = M//5 
        return np.mean(self.payoff(S[:,-1]) * prob * sample[:,0] / (self.spot * self.sigma * dt**0.5)) * self.df
#         return np.mean(self.payoff(S[:,-1]) * prob * sample[:,k-1] / (self.spot * self.sigma * (dt*k)**0.5)) * self.df
    
    def CEMC_gamma(self, N, M):
        dt = self.T/M
        sample = self._sample(N, M, "normal")
        S = _stock_price(self.T, self.r, self.sigma, self.spot, N, M, sample)
        prob = _CE_mc_sample(self.T, self.sigma, N, M, self.b, S, self.type)
        if prob is None:
            raise MyException("Error: Only for knock-out type, use in-out parity to calculate the price") 
#         k = M//5 
        score1 = (sample[:,0]**2-1)/(self.spot**2 * self.sigma**2 * dt)
        score2 = sample[:,0] / (self.spot**2 * self.sigma * dt**0.5)
#         score1 = (sample[:,k-1]**2-1)/(self.spot**2 * self.sigma**2 * (dt*k))
#         score2 = sample[:,k-1] / (self.spot**2 * self.sigma * (dt*k)**0.5)
        score = score1 - score2
        return np.mean(self.payoff(S[:,-1]) * prob * score) * self.df
    
    def CEMC_vega(self, N, M):
        dt = self.T/M
        sample = self._sample(N, M, "normal")
        S = _stock_price(self.T, self.r, self.sigma, self.spot, N, M, sample)
        prob = _CE_mc_sample(self.T, self.sigma, N, M, self.b, S, self.type)
        if prob is None:
            raise MyException("Error: Only for knock-out type, use in-out parity to calculate the price") 
        score = (sample[:,:-1]**2-1)/self.sigma -  sample[:,:-1]*dt**0.5
        score = np.sum(score, axis = 1)
        return np.mean(self.payoff(S[:,-1]) * prob * score) * self.df       
        
class MyException(Exception):
    pass    