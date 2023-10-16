clc
% step 1
portfolio = 0.125 * ones(1,8);
x = 0.125 * ones(1,8);
for k = 0 : 100000
    scenario = mod(k,7) + 1;
    % step 2 Solve Q(x,w)
    u = [0 0 0 0];
    for t = 1 : 4
        if (R{t}(11) - returnrates{t}(scenario,:) * x') > 0
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
    % step 3 update x
    x_new = projunitsimplex(x' - 1/(k + 1) * (c' + g'));
    x = x_new';
    portfolio = [portfolio ; x];
    if k > 1
        if max([norm(portfolio(k+2,:) - portfolio(k+1,:)), norm(portfolio(k+2,:) - portfolio(k,:)), norm(portfolio(k+2,:) - portfolio(k-1,:))]) < 1.0000e-06
            break
        end
    end
end