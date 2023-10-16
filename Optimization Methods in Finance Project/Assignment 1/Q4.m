% Q4
clc
clear
M = csvread('indices.csv', 1, 1);
dim = size(M);
M = flip(M);
return_rate=[];
tem_return = [];
assets = {'Dow-Jones','FTSE','DAX','CAC','Nikkei','HSI','BOVESPA','Gold'};
for i =1:dim(2)
    for j = 1:(dim(1)-1)
        tem_return = [tem_return (M(j+1,i)-M(j,i))/M(j,i)];
        % Calculate the return rate
    end
    tem_return = tem_return';
    return_rate=[return_rate tem_return];
    tem_return = [];
end
return_rate = 1 + return_rate;

% Questoin 1
Cov= cov(return_rate)*95/96;
var_Nikkei = Cov(5,5);
e=[1 1 1 1 1 1 1 1];
geo = geo_mean(return_rate);
ari = mean(return_rate);


% Question 2
% Use arithmetic mean as the expected return

opt_ari=[];
expected_return_ari=[];
risk_std_ari = [];

for R=1:0.0001:1.006
cvx_begin quiet
variables x(8)
minimize(x'*Cov*x)
subject to 
e*x==1;
x>=0;
ari*x >= R;
cvx_end
if isnan(x)
    R_largest_ari = R-0.0001;
    break
end
opt_ari=[opt_ari x];
expected_return_ari=[expected_return_ari ari*x];
risk_std_ari = [risk_std_ari sqrt(x'*Cov*x)];
end

R=1:0.0001:1.006;
diff_opt_ari = diff(opt_ari,1,2);
for i =size(diff_opt_ari,2):-1:1
if abs(diff_opt_ari(:,i))<1e-3
    i=i+1;
break
end
end
R_smallest_ari = R(i);

i_smallest_ari=find(R==R_smallest_ari);
i_largest_ari=find(R==R_largest_ari);
% plot
figure;
plot(risk_std_ari(1,i_smallest_ari:end),expected_return_ari(1,i_smallest_ari:end),'b+-','LineWidth',2);
title('Efficient Frontier(arithmetic mean)');
xlabel('Standard deviation');
ylabel(' Expected return');

figure;
area(R(1,i_smallest_ari:i_largest_ari),opt_ari(:,i_smallest_ari:end)')
title('Composition of Efficient Portfolios(arithmetic mean)');
xlabel('Target Return');
ylabel(' Investement in each asset');
legend(assets);

figure;
area(expected_return_ari(1,i_smallest_ari:end),opt_ari(:,i_smallest_ari:end)');
title('Composition of Efficient Portfolios(arithmetic mean)');
xlabel(' Expected return');
ylabel(' Investement in each asset');
legend(assets);


% Use geometric mean as the expected return
opt_geo=[];
expected_return_geo=[];
risk_std_geo = [];
for R=1:0.0001:1.006
cvx_begin quiet
variables x(8)
minimize(x'*Cov*x)
subject to 
e*x==1;
x>=0;
geo*x >= R;
cvx_end
if isnan(x)
    R_largest_geo = R-0.0001;
    break
end
opt_geo=[opt_geo x];
expected_return_geo=[expected_return_geo geo*x];
risk_std_geo = [risk_std_geo sqrt(x'*Cov*x)];
end

R=1:0.0001:1.006;
diff_opt_geo = diff(opt_geo,1,2);
for i =size(diff_opt_geo,2):-1:1
if abs(diff_opt_geo(:,i))<1e-3
    i=i+1;
break
end
end
R_smallest_geo = R(i);

i_smallest_geo=find(R==R_smallest_geo);



i_largest_geo=find(R==R_largest_geo);
% plot
figure;
plot(risk_std_geo(1,i_smallest_geo:end),expected_return_geo(1,i_smallest_geo:end),'b+-','LineWidth',2);
title('Efficient Frontier(geometric mean)');
xlabel('Standard deviation');
ylabel(' Expected return');

figure;
area(R(1,i_smallest_geo:i_largest_geo),opt_geo(:,i_smallest_geo:end)')
title('Composition of Efficient Portfolios(geometric mean)');
xlabel('Target Return');
ylabel(' Investement in each asset');
legend(assets);

figure;
area(expected_return_geo(1,i_smallest_geo:end),opt_geo(:,i_smallest_geo:end)');
title('Composition of Efficient Portfolios(geometric mean)');
xlabel(' Expected return');
ylabel(' Investement in each asset');
legend(assets);

% Question 3
cvx_begin quiet
variables x(8)
minimize(x'*Cov*x)
subject to 
e*x==1;
x>=0;
geo*x >= 1.0024;
cvx_end
x

% 4
dim = size(return_rate);
mad = zeros(size(return_rate));
for i = 1:dim(1)
    mad(i,:) = (return_rate(i,:)-geo);
end
cvx_begin quiet
variables x(8)
minimize(ones(1,96)*abs(mad*x)/96)
subject to

e*x==1;
x>=0;
geo*x >= 1.0024;
cvx_end
optimal = x;


% Q5
standard_deviation = sqrt(optimal'*Cov*optimal)
% variance = mean((return_rate*optimal-mean(return_rate)*optimal).^2)
mean_absolute_deviation = mean(abs(mad*optimal))
semi_deviation = sqrt(mean(max(0,(return_rate*optimal-geo*optimal)).^2))