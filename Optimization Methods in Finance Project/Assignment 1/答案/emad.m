function emad(mu,ret,prob,Target,delta,B,L,assets)
% Minimize E[X] + delta*MAD[X]

n = size(ret,2);

X = []; ExpRet = []; EMAD = []; 

RR = mu - ret;

for k = 1:length(Target)
    cvx_begin quiet % The keyword quiet silences CVX output to screen
      variable x(n);  % proportion of investment in each asset
      minimize( L - B*mu*x + delta*B*prob*abs(RR*x) );
      mu*x >= Target(k);
      ones(1,n)*x == 1;
      x >= 0;
    cvx_end
    
    X = [X x];
    ExpRet = [ExpRet, mu*x];
    obj = (L - B*mu*x + delta*B*prob*abs(RR*x))/B;  % Must scale it, so divide by B
    EMAD = [EMAD, obj]; 
end

frontierarea(ExpRet,EMAD,X,Target,{'E + delta*MAD','% Target Return'},assets);

end