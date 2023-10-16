function cvarportfolio(mu,ret,prob,Target,beta,B,L,assets)
% Minimize CVaR of shortfall

n = size(ret,2);  
m = size(ret,1);

X = []; ExpRet = []; Cvar = []; 

for k = 1:length(Target)
    cvx_begin quiet % The keyword quiet silences CVX output to screen
      variable x(n);  % proportion of investment in each asset
      variable z(m);
      variable t;
      minimize(t + prob*z/(1-beta));
      z + ones(m,1)*t + B*ret*x >= L;
      mu*x >= Target(k);
      ones(1,n)*x == 1;
      x >= 0;
      z >= 0;
    cvx_end
    
    X = [X x];
    ExpRet = [ExpRet, mu*x];
    obj = t + prob*z/(1-beta);
    Cvar = [Cvar, obj]; 
end

frontierarea(ExpRet,Cvar,X,Target,{'CVaR','% Target Return'},assets);

X(:,13)
Target(:,13)

sfall = max(0, L - B*ret*X); % row = scenario, column = portfolio for target
expsfall = prob * sfall
end