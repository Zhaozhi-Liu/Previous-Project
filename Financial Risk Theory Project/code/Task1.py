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
    
    var_level = q*100
    var = np.percentile(sample, var_level)
    cvar = sample[sample >= var].mean()
    sgld_var = theta
    sgld_cvar = np.mean(np.array([max(sample[i]-theta,0) for i in range(n)])*(1/(1-q))+theta)+gamma*theta**2
    return sgld_var,sgld_cvar,var,cvar
# q = 0.95 normal distribution-------------------------------------------------
sample1 = rng.normal(size=n)
sgld_var,sgld_cvar,var,cvar = secB(sample1,0.95)
sgld_var,sgld_cvar,var,cvar
#------------------------------------------------------------------------------
sample2 = rng.normal(loc=1,scale=2,size=n)
sgld_var,sgld_cvar,var,cvar = secB(sample2,0.95)
sgld_var,sgld_cvar,var,cvar
#------------------------------------------------------------------------------
sample3 = rng.normal(loc=3,scale=5,size=n)
sgld_var,sgld_cvar,var,cvar = secB(sample3,0.95)
sgld_var,sgld_cvar,var,cvar
#------------------------------------------------------------------------------

# q=0.99 normal distribution---------------------------------------------------
sample4 = rng.normal(size=n)
sgld_var,sgld_cvar,var,cvar = secB(sample4,0.99)
sgld_var,sgld_cvar,var,cvar
#------------------------------------------------------------------------------
sample5 = rng.normal(loc=1,scale=2,size=n)
sgld_var,sgld_cvar,var,cvar = secB(sample5,0.99)
sgld_var,sgld_cvar,var,cvar
#------------------------------------------------------------------------------
sample6 = rng.normal(loc=3,scale=5,size=n)
sgld_var,sgld_cvar,var,cvar = secB(sample6,0.99)
sgld_var,sgld_cvar,var,cvar
#------------------------------------------------------------------------------

# q=0.95 t distribution--------------------------------------------------------
sample7 = rng.standard_t(df = 10,size = n)
sgld_var,sgld_cvar,var,cvar = secB(sample7,0.95)
sgld_var,sgld_cvar,var,cvar
#------------------------------------------------------------------------------
sample8 = rng.standard_t(df = 7,size = n)
sgld_var,sgld_cvar,var,cvar = secB(sample8,0.95)
sgld_var,sgld_cvar,var,cvar
#------------------------------------------------------------------------------
sample9 = rng.standard_t(df = 3,size = n)
sgld_var,sgld_cvar,var,cvar = secB(sample9,0.95)
sgld_var,sgld_cvar,var,cvar
#------------------------------------------------------------------------------

# q=0.99 t distribution--------------------------------------------------------
sample10 = rng.standard_t(df = 10,size = n)
sgld_var,sgld_cvar,var,cvar = secB(sample10,0.99)
sgld_var,sgld_cvar,var,cvar
#------------------------------------------------------------------------------
sample11 = rng.standard_t(df = 7,size = n)
sgld_var,sgld_cvar,var,cvar = secB(sample11,0.99)
sgld_var,sgld_cvar,var,cvar
#------------------------------------------------------------------------------
sample12 = rng.standard_t(df = 3,size = n)
sgld_var,sgld_cvar,var,cvar = secB(sample12,0.99)
sgld_var,sgld_cvar,var,cvar



A=[]
i=5
for i in range(5):
    A=np.append(np.array(A),np.array([i,i]),axis =0)
A=np.reshape(A,[5,2])
