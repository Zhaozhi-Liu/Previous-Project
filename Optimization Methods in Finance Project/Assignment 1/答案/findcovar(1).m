function Sigma = findcovar(ret, prob)
% Finds covariance of a returns matrix with assets in columns
% and rows for historical values
% prob = probabilities for scenarios past or future
% If uniform probabilities, then scenarios are equally likely 
% and so mean becomes arithmetic mean

mu = prob * ret;   % mean of each column
n = size(ret,2);  
m = size(ret,1);
Sigma = zeros(n,n);

% Compute Covariance matrix
% First compute the lower triangular part
for i=1:n
    for j=1:i
        diff = [ret(:,i) - mu(i), ret(:,j) - mu(j)];
        vec = diff(:,1).*diff(:,2) ;
        Sigma(i,j) = prob * vec ;
    end
end
% Now flip the matrix and add
Sigma = Sigma + triu(Sigma',1);