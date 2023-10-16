function X = mad(mu,ret,prob,Target,assets)
% Minimize MAD[X]

n = size(ret,2);

X = []; ExpRet = []; MAD = []; 

RR = ret - mu;

for k = 1:length(Target)
    cvx_begin quiet % The keyword quiet silences CVX output to screen
      variable x(n);  % proportion of investment in each asset
      minimize( prob*abs(RR*x) );
      mu*x >= Target(k);
      ones(1,n)*x == 1;
      x >= 0;
    cvx_end
    
    X = [X x];
    ExpRet = [ExpRet, mu*x];
    obj = prob*abs(RR*x); 
    MAD = [MAD, obj]; 
end
X
%frontierarea(ExpRet,MAD,X,Target,{'MAD','% Target Return'},assets);
MAD
end