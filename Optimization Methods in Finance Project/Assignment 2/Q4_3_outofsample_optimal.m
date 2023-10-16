% out of sample
Q1_raw = csvread('Q1_raw.csv',1,2);
Q2_raw = csvread('Q2_raw.csv',1,2);
Q3_raw = csvread('Q3_raw.csv',1,2);
Q4_raw = csvread('Q4_raw.csv',1,2);
Q_raw = {Q1_raw Q2_raw Q3_raw Q4_raw};
outsample_scenario = min([size(Q1_raw,1) size(Q2_raw,1) size(Q3_raw,1) size(Q4_raw,1)]);
% The first algorithm
portfolio = 0.125 * ones(1,8);
x = 0.125 * ones(1,8);
for k = 0 : 10000
    scenario = mod(k,outsample_scenario) + 1;
    % step 2 Solve Q(x,w)
    u = [0 0 0 0];
    for t = 1 : 4
        if (R{t}(11) - Q_raw{t}(scenario,:) * x') > 0
            u(t) = costs(t);
        else
            u(t) = 0;
        end
    end
    P = zeros(1,8);
    for i = 1 : 8
       P(i) = -[Q_raw{1}(scenario,i) Q_raw{2}(scenario,i) Q_raw{3}(scenario,i) Q_raw{4}(scenario,i)] * u';
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
x_1_outofsample = x_new';

% The second algorithm 
portfolio = 0.125 * ones(1,8);
x = 0.125 * ones(1,8);
N = outsample_scenario;
for k = 0 : 10000
    % step 2 Solve Q(x,w)
    gradient = zeros(1,8);
    for scenario = 1 : outsample_scenario
        y = [0 0 0 0];
        y(1) = max([0, R{1}(11)-Q_raw{1}(scenario,:)*x']);
        y(2) = max([0, R{2}(11)-Q_raw{2}(scenario,:)*x']);
        y(3) = max([0, R{3}(11)-Q_raw{3}(scenario,:)*x']);
        y(4) = max([0, R{4}(11)-Q_raw{4}(scenario,:)*x']);
        R_matrix = [Q_raw{1}(scenario,:) ; Q_raw{2}(scenario,:) ; Q_raw{3}(scenario,:) ; Q_raw{4}(scenario,:)];
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
x_2_outofsample = x_new';