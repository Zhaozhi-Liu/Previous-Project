assets = {'S&P100', 'S&P500', 'S&P600', 'Dow', 'NASDAQ', 'Russell 2000', 'Barrons', 'Wilshire'};
% Algorithm 1
subplot(2,2,1)
area(R{1},opt_1')
title('Composition of Efficient Portfolios(Q1)');
xlabel('Target Return');
ylabel('Investement in each asset');
legend(assets);

subplot(2,2,2)
area(R{2},opt_1')
title('Composition of Efficient Portfolios(Q2)');
xlabel('Target Return');
ylabel('Investement in each asset');
legend(assets);

subplot(2,2,3)
area(R{3},opt_1')
title('Composition of Efficient Portfolios(Q3)');
xlabel('Target Return');
ylabel('Investement in each asset');
legend(assets);

subplot(2,2,4)
area(R{4},opt_1')
title('Composition of Efficient Portfolios(Q4)');
xlabel('Target Return');
ylabel('Investement in each asset');
legend(assets);
%--------------------------------------------------------------------------
subplot(2,2,1)
area(Q1_expected_alg1,opt_1')
title('Composition of Efficient Portfolios(Q1)');
xlabel('Expected Return');
ylabel('Investement in each asset');
legend(assets);

subplot(2,2,2)
area(Q2_expected_alg1,opt_1')
title('Composition of Efficient Portfolios(Q2)');
xlabel('Expected Return');
ylabel('Investement in each asset');
legend(assets);

subplot(2,2,3)
area(Q3_expected_alg1,opt_1')
title('Composition of Efficient Portfolios(Q3)');
xlabel('Expected Return');
ylabel('Investement in each asset');
legend(assets);

subplot(2,2,4)
area(Q4_expected_alg1,opt_1')
title('Composition of Efficient Portfolios(Q4)');
xlabel('Expected Return');
ylabel('Investement in each asset');
legend(assets);
%--------------------------------------------------------------------------
%Algorithm 2
subplot(2,2,1)
area(R{1},opt_2')
title('Composition of Efficient Portfolios(Q1)');
xlabel('Target Return');
ylabel('Investement in each asset');
legend(assets);

subplot(2,2,2)
area(R{2},opt_2')
title('Composition of Efficient Portfolios(Q2)');
xlabel('Target Return');
ylabel('Investement in each asset');
legend(assets);

subplot(2,2,3)
area(R{3},opt_2')
title('Composition of Efficient Portfolios(Q3)');
xlabel('Target Return');
ylabel('Investement in each asset');
legend(assets);

subplot(2,2,4)
area(R{4},opt_2')
title('Composition of Efficient Portfolios(Q4)');
xlabel('Target Return');
ylabel('Investement in each asset');
legend(assets);
%--------------------------------------------------------------------------
subplot(2,2,1)
area(Q1_expected_alg2,opt_2')
title('Composition of Efficient Portfolios(Q1)');
xlabel('Expected Return');
ylabel('Investement in each asset');
legend(assets);

subplot(2,2,2)
area(Q2_expected_alg2,opt_2')
title('Composition of Efficient Portfolios(Q2)');
xlabel('Expected Return');
ylabel('Investement in each asset');
legend(assets);

subplot(2,2,3)
area(Q3_expected_alg2,opt_2')
title('Composition of Efficient Portfolios(Q3)');
xlabel('Expected Return');
ylabel('Investement in each asset');
legend(assets);

subplot(2,2,4)
area(Q4_expected_alg2,opt_2')
title('Composition of Efficient Portfolios(Q4)');
xlabel('Expected Return');
ylabel('Investement in each asset');
legend(assets);