import numpy as np
import math
rng = np.random.default_rng()

def SecBTtask2(asset1,asset2,asset3,q=0.95):
    beta = 10**8
    gamma = 10**(-8)
    lamda = 10**(-4)
    n = 10**6
    xi = rng.normal(size = (n,4))
    theta_hat = np.array([0,1/3,1/3,1/3])
    H_hat = np.array([0,0,0,0])
    for i in range(n):
        sum_g = math.exp(theta_hat[1])+math.exp(theta_hat[2])+math.exp(theta_hat[3])
        g_1_omega = math.exp(theta_hat[1])/sum_g
        g_2_omega = math.exp(theta_hat[2])/sum_g
        g_3_omega = math.exp(theta_hat[3])/sum_g
        characterfun = g_1_omega*asset1[i] + g_2_omega*asset2[i]+ g_3_omega*asset3[i]
        g_hat_1 = (math.exp(theta_hat[1])*(math.exp(theta_hat[2])+math.exp(theta_hat[3])))/(sum_g**2)*asset1[i]-\
            math.exp(theta_hat[1])*math.exp(theta_hat[2])/(sum_g**2)*asset2[i]-\
                math.exp(theta_hat[1])*math.exp(theta_hat[3])/(sum_g**2)*asset3[i]
        g_hat_2 = -math.exp(theta_hat[1])*math.exp(theta_hat[2])/(sum_g**2)*asset1[i]+\
            (math.exp(theta_hat[2])*(math.exp(theta_hat[1])+math.exp(theta_hat[3])))/(sum_g**2)*asset2[i]-\
                math.exp(theta_hat[2])*math.exp(theta_hat[3])/(sum_g**2)*asset3[i]
        g_hat_3 = -math.exp(theta_hat[1])*math.exp(theta_hat[3])/(sum_g**2)*asset1[i]-\
            math.exp(theta_hat[3])*math.exp(theta_hat[2])/(sum_g**2)*asset2[i]+\
                (math.exp(theta_hat[3])*(math.exp(theta_hat[1])+math.exp(theta_hat[2])))/(sum_g**2)*asset3[i]
        if characterfun >= theta_hat[0]:
            H_hat[0] = 1-1/(1-q) + 2*gamma*theta_hat[0]
            H_hat[1] = g_hat_1/(1-q) + 2*gamma*theta_hat[1]
            H_hat[2] = g_hat_2/(1-q) + 2*gamma*theta_hat[2]
            H_hat[3] = g_hat_3/(1-q) + 2*gamma*theta_hat[3]
        else:
            H_hat[0] = 1+2*gamma*theta_hat[0]
            H_hat[1] = 2*gamma*theta_hat[1]
            H_hat[2] = 2*gamma*theta_hat[2]
            H_hat[3] = 2*gamma*theta_hat[3]
        theta_hat = theta_hat - lamda * H_hat + (2*beta**(-1)*lamda)**(1/2)*xi[i]
    sum_g = math.exp(theta_hat[1])+math.exp(theta_hat[2])+math.exp(theta_hat[3])
    g_1_omega = math.exp(theta_hat[1])/sum_g
    g_2_omega = math.exp(theta_hat[2])/sum_g
    g_3_omega = math.exp(theta_hat[3])/sum_g
    sample = g_1_omega*asset1+g_2_omega*asset2+g_3_omega*asset3
    var = np.quantile(sample,q)
    cvar = sample[sample >= var].mean()
    return var,cvar,g_1_omega,g_2_omega,g_3_omega



def get_approx(n=10):
    approx = []
    for k in range(n):
        asset1=rng.normal(loc = 500, scale = 1, size = 10**6)
        asset2=rng.normal(loc = 0,scale = 10**3, size = 10**6)
        asset3=rng.normal(loc = 0,scale = 10**(-2), size = 10**6)# need changing with respect to different parameters of dist.
        [var,cvar,g_1_omega,g_2_omega,g_3_omega] = SecBTtask2(asset1,asset2,asset3,q=0.95)
        approx = np.append(np.array(approx),[var,cvar,g_1_omega,g_2_omega,g_3_omega],axis = 0)
    approx = np.reshape(approx,[n,5])
    var,cvar,g_1_omega,g_2_omega,g_3_omega = np.mean(approx,axis = 0)
    return var,cvar,g_1_omega,g_2_omega,g_3_omega
var,cvar,g_1_omega,g_2_omega,g_3_omega=get_approx()
print(f'The SGLD estimated VaR is {round(var,4)}')
print(f'The SGLD estimated CVaR is {round(cvar,4)}')
print(f'The estimated weights is {round(g_1_omega,4)}, {round(g_2_omega,4)}, {round(g_3_omega,4)}')













def SecBTtask2_check(asset1,asset2,q=0.95):
    beta = 10**8
    gamma = 10**(-8)
    lamda = 10**(-4)
    n = 10**6
    xi = rng.normal(size = (n,3))
    theta_hat = np.array([0,1/2,1/2])
    H_hat = np.array([0,0,0])
    for i in range(n):
        sum_g = math.exp(theta_hat[1])+math.exp(theta_hat[2])
        g_1_omega = math.exp(theta_hat[1])/sum_g
        g_2_omega = math.exp(theta_hat[2])/sum_g
        characterfun = g_1_omega*asset1[i] + g_2_omega*asset2[i]
        g_hat_1 = ((math.exp(theta_hat[1])*(math.exp(theta_hat[2])))/(sum_g**2))*asset1[i]-\
            (math.exp(theta_hat[1])*math.exp(theta_hat[2])/(sum_g**2))*asset2[i]
        g_hat_2 = -(math.exp(theta_hat[1])*math.exp(theta_hat[2])/(sum_g**2))*asset1[i]+\
            ((math.exp(theta_hat[2])*(math.exp(theta_hat[1])))/(sum_g**2))*asset2[i]
        if characterfun >= theta_hat[0]:
            H_hat[0] = 1-1/(1-q) + 2*gamma*theta_hat[0]
            H_hat[1] = g_hat_1/(1-q) + 2*gamma*theta_hat[1]
            H_hat[2] = g_hat_2/(1-q) + 2*gamma*theta_hat[2]
        else:
            H_hat[0] = 1+2*gamma*theta_hat[0]
            H_hat[1] = 2*gamma*theta_hat[1]
            H_hat[2] = 2*gamma*theta_hat[2]
        theta_hat = theta_hat - lamda * H_hat + (2*beta**(-1)*lamda)**(1/2)*xi[i]
    sum_g = math.exp(theta_hat[1])+math.exp(theta_hat[2])
    g_1_omega = math.exp(theta_hat[1])/sum_g
    g_2_omega = math.exp(theta_hat[2])/sum_g
    sample = g_1_omega*asset1+g_2_omega*asset2
    var = np.quantile(sample,0.95)
    cvar = sample[sample >= var].mean()
    return var,cvar,g_1_omega, g_2_omega


asset1 = rng.normal(1,2,size = 10**6)
asset2 = rng.logistic(0,1,size = 10**6)
SecBTtask2_check(asset1,asset2)
cvar = []
even_space = np.linspace(0,1,100)
for i in range(100):
    portfolio = even_space[i]*asset1+(1-even_space[i])*asset2
    var = np.quantile(portfolio,0.95)
    cvar.append(portfolio[portfolio >= var].mean())
cvar_true = np.min(cvar)
index = np.argmin(cvar)
var = np.quantile(even_space[index]*asset1+(1-even_space[index])*asset2,0.95)
w1 = even_space[index]
w2 = 1 - even_space[index]