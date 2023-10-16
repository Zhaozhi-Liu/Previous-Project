clc
% step 1 
portfolio = 0.125 * ones(1,8);
x = 0.125 * ones(1,8);
N = 7;
for k = 0 : 100000
    % step 2 Solve Q(x,w)
    gradient = zeros(1,8);
    for scenario = 1 : N
        y = [0 0 0 0];
        y(1) = max([0, R{1}(11)-returnrates{1}(scenario,:)*x']);
        y(2) = max([0, R{2}(11)-returnrates{2}(scenario,:)*x']);
        y(3) = max([0, R{3}(11)-returnrates{3}(scenario,:)*x']);
        y(4) = max([0, R{4}(11)-returnrates{4}(scenario,:)*x']);
        R_matrix = [returnrates{1}(scenario,:) ; returnrates{2}(scenario,:) ; returnrates{3}(scenario,:) ; returnrates{4}(scenario,:)];
        gradient = gradient + (-costs.*sign(y) * R_matrix);
    end
    g = 1 / N * gradient;
    % step 3 update x
    x_new = projunitsimplex(x' - 1/(k + 1) * (c' + g'));
    x = x_new';
    portfolio = [portfolio ; x];
    if k > 1
        if max([norm(portfolio(k+2,:) - portfolio(k+1,:)) norm(portfolio(k+2,:) - portfolio(k,:)) norm(portfolio(k+2,:) - portfolio(k-1,:))]) < 1.0000e-06
            break
        end
    end
end