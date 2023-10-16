subplot(2,2,1)
hold on 
plot(Q1_std_alg1,Q1_expected_alg1,'b+-','LineWidth',2);
plot(Q1_std_alg2,Q1_expected_alg2,'r*-','LineWidth',2);
xlabel('Standard deviation');
ylabel('Expected return');
legend('Algorithm 1','Algorithm2')
title('Efficient Frontier(Q1 4.1 vs 4.2)');
hold off 

subplot(2,2,2)
hold on 
plot(Q2_std_alg1,Q2_expected_alg1,'b+-','LineWidth',2);
plot(Q2_std_alg2,Q2_expected_alg2,'r*-','LineWidth',2);
xlabel('Standard deviation');
ylabel('Expected return');
legend('Algorithm 1','Algorithm2')
title('Efficient Frontier(Q2 4.1 vs 4.2)');
hold off 

subplot(2,2,3)
hold on 
plot(Q3_std_alg1,Q3_expected_alg1,'b+-','LineWidth',2);
plot(Q3_std_alg2,Q3_expected_alg2,'r*-','LineWidth',2);
xlabel('Standard deviation');
ylabel('Expected return');
legend('Algorithm 1','Algorithm2')
title('Efficient Frontier(Q3 4.1 vs 4.2)');
hold off 

subplot(2,2,4)
hold on 
plot(Q4_std_alg1,Q4_expected_alg1,'b+-','LineWidth',2);
plot(Q4_std_alg2,Q4_expected_alg2,'r*-','LineWidth',2);
xlabel('Standard deviation');
ylabel('Expected return');
legend('Algorithm 1','Algorithm2')
title('Efficient Frontier(Q4 4.1 vs 4.2)');
hold off 