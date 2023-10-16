function frontierarea(ExpRet,stdev,X,xaxis,x_label,assets)

PercRet = 100*([ExpRet;xaxis]- 1);

figure;
plot(stdev,PercRet(1,:),'b+-','LineWidth',2);
title('Efficient Frontier');
xlabel(char(x_label(1)));
ylabel('% Expected return');

figure;
area(PercRet(2,:),X');
title('Composition of Efficient Portfolios');
xlabel(char(x_label(2)));
ylabel('% Investement in each asset');
legend(assets);

figure;
area(PercRet(1,:),X');
title('Composition of Efficient Portfolios');
xlabel('% Expected return');
ylabel('% Investement in each asset');
legend(assets);