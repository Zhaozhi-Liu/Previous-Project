% Algorithm 1
opt_1 = [];
Q1_expected = [];
Q2_expected = [];
Q3_expected = [];
Q4_expected = [];
Q1_std = [];
Q2_std = [];
Q3_std = [];
Q4_std = [];
for target = 1 : 21
    portfolio = 0.125 * ones(1,8);
    x = 0.125 * ones(1,8);
    for k = 0 : 10000
        scenario = mod(k,7) + 1;
        u = [0 0 0 0];
        for t = 1 : 4
            if (R{t}(target) - returnrates{t}(scenario,:) * x') > 0
                u(t) = costs(t);
            else
                u(t) = 0;
            end
        end
        P = zeros(1,8);
        for i = 1 : 8
            P(i) = -[returnrates{1}(scenario,i) returnrates{2}(scenario,i) returnrates{3}(scenario,i) returnrates{4}(scenario,i)] * u';
        end
        g = P;
        x_new = projunitsimplex(x' - 1/(k + 1) * (c' + g'));
        x = x_new';
        portfolio = [portfolio ; x];
        if k > 1
            if max([norm(portfolio(k+2,:) - portfolio(k+1,:)) norm(portfolio(k+2,:) - portfolio(k,:)) norm(portfolio(k+2,:) - portfolio(k-1,:))]) < 1.0000e-06
                break
            end
        end
    end
    opt_1 = [opt_1 x_new];
    Q1_expected = [Q1_expected mu_1 * x_new];
    Q2_expected = [Q2_expected mu_2 * x_new];
    Q3_expected = [Q3_expected mu_3 * x_new];
    Q4_expected = [Q4_expected mu_4 * x_new];
    Q1_std = [Q1_std sqrt(x_new'*Sigma_1*x_new)];
    Q2_std = [Q2_std sqrt(x_new'*Sigma_2*x_new)];
    Q3_std = [Q3_std sqrt(x_new'*Sigma_3*x_new)];
    Q4_std = [Q4_std sqrt(x_new'*Sigma_4*x_new)];
end


subplot(2,2,1)
plot(Q1_std,Q1_expected,'b+-','LineWidth',2);
title('Efficient Frontier(Q1 4.1)');
xlabel('Standard deviation');
ylabel('Expected return');

subplot(2,2,2)
plot(Q2_std,Q2_expected,'r+-','LineWidth',2);
title('Efficient Frontier(Q2 4.1)');
xlabel('Standard deviation');
ylabel('Expected return');

subplot(2,2,3)
plot(Q3_std,Q3_expected,'g+-','LineWidth',2);
title('Efficient Frontier(Q3 4.1)');
xlabel('Standard deviation');
ylabel('Expected return');

subplot(2,2,4)
plot(Q4_std,Q4_expected,'k+-','LineWidth',2);
title('Efficient Frontier(Q4 4.1)');
xlabel('Standard deviation');
ylabel('Expected return');

Q1_expected_alg1 = Q1_expected;
Q1_std_alg1 = Q1_std;

Q2_expected_alg1 = Q2_expected;
Q2_std_alg1 = Q2_std;

Q3_expected_alg1 = Q3_expected;
Q3_std_alg1 = Q3_std;

Q4_expected_alg1 = Q4_expected;
Q4_std_alg1 = Q4_std;