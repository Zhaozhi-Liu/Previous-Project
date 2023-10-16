import numpy as np
rng = np.random.default_rng()
n=10**6
def secB(sample,q,n = 10**6):
    beta = 10**8
    gamma = 10**(-8)
    lamda = 10**(-4)
    xi = rng.normal(size = n)
    theta = 0
    for i in range(n):
        if sample[i]<theta:
            H = -(q/(1-q)) + 1/(1-q) + 2*gamma*theta
        else:
            H = -(q/(1-q)) + 2*gamma*theta
        theta = theta - lamda * H + (2*beta**(-1)*lamda)**(1/2)*xi[i]
    
    sgld_var = theta
    sgld_cvar = np.mean(np.array([max(sample[i]-theta,0) for i in range(n)])*(1/(1-q))+theta)+gamma*theta**2
    return sgld_var,sgld_cvar

def get_approx(q,n=50):
    approx = []
    for i in range(n):
        sample = rng.normal(3,5,size = 10**6)# need changing with respect to different parameters of dist.
        [sgld_var,sgld_cvar] = secB(sample,q)
        approx = np.append(np.array(approx),[sgld_var,sgld_cvar],axis = 0)
    approx = np.reshape(approx,[n,2])
    sgld_var,sgld_cvar = np.mean(approx,axis = 0)
    sgld_var_std,sgld_cvar_std = np.std(approx,axis = 0)
    return sgld_var,sgld_var_std,sgld_cvar,sgld_cvar_std

sgld_var,sgld_var_std,sgld_cvar,sgld_cvar_std=get_approx(0.99)
print(f'The SGLD estimated VaR is {round(sgld_var,4)}, the standard deviation is {round(sgld_var_std,4)}')
print(f'The SGLD estimated CVaR is {round(sgld_cvar,4)}, the standard deviation is {round(sgld_cvar_std,6)}')

sample = rng.standard_t(df = 10,size = 10**6)