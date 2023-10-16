clc
clear
% Question 3.1 & 3.2
Q1_returnrates = csvread('Q1_returnrates.csv',1,1);
Q2_returnrates = csvread('Q2_returnrates.csv',1,1);
Q3_returnrates = csvread('Q3_returnrates.csv',1,1);
Q4_returnrates = csvread('Q4_returnrates.csv',1,1);
mu_1 = geo_mean(Q1_returnrates);
mu_2 = geo_mean(Q2_returnrates);
mu_3 = geo_mean(Q3_returnrates);
mu_4 = geo_mean(Q4_returnrates);
Sigma_1 = cov(Q1_returnrates)*6/7;
Sigma_2 = cov(Q2_returnrates)*6/7;
Sigma_3 = cov(Q3_returnrates)*6/7;
Sigma_4 = cov(Q4_returnrates)*6/7;
% Generate target return rates.
% Question 3.3
R_1 = [];
R_2 = [];
R_3 = [];
R_4 = [];
for delta = 0:0.05:1
    r_1 = min(mu_1) + delta * (max(mu_1) - min(mu_1));
    r_2 = min(mu_2) + delta * (max(mu_2) - min(mu_2));
    r_3 = min(mu_3) + delta * (max(mu_3) - min(mu_3));
    r_4 = min(mu_4) + delta * (max(mu_4) - min(mu_4));
    R_1 = [R_1 r_1];
    R_2 = [R_2 r_2];
    R_3 = [R_3 r_3];
    R_4 = [R_4 r_4];
end

costs = [1.3 2.5 1.75 3.25];
returnrates = {Q1_returnrates,Q2_returnrates,Q3_returnrates,Q4_returnrates};
R = {R_1,R_2,R_3,R_4};
c = [0.45 1.15 0.65 0.8 1.25 1.1 0.9 0.7]*0.01;
% step 1 
portfolio = 0.125 * ones(1,8);
x = 0.125 * ones(1,8);
