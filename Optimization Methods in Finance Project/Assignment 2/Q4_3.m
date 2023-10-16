x_1 = [0.1799 0.1136 0.1332 0.1447 0.1052 0.0733 0.0976 0.1525];
x_2 = [0.1817 0.1160 0.1311 0.1483 0.1025 0.0710 0.0948 0.1545];
Q_1 = [];
Q_2 = [];
for scenario = 1 : 7
    y_1 = [0 0 0 0];
    y_2 = [0 0 0 0];
    y_1(1) = max([0, R{1}(11)-returnrates{1}(scenario,:)*x_1']);
    y_1(2) = max([0, R{2}(11)-returnrates{2}(scenario,:)*x_1']);
    y_1(3) = max([0, R{3}(11)-returnrates{3}(scenario,:)*x_1']);
    y_1(4) = max([0, R{4}(11)-returnrates{4}(scenario,:)*x_1']);
    y_2(1) = max([0, R{1}(11)-returnrates{1}(scenario,:)*x_2']);
    y_2(2) = max([0, R{2}(11)-returnrates{2}(scenario,:)*x_2']);
    y_2(3) = max([0, R{3}(11)-returnrates{3}(scenario,:)*x_2']);
    y_2(4) = max([0, R{4}(11)-returnrates{4}(scenario,:)*x_2']);
    Q_1 = [Q_1 costs * y_1'];
    Q_2 = [Q_2 costs * y_2'];
end
z_insample_1 = c * x_1' + sum(1/7 * Q_1);
z_insample_2 = c * x_2' + sum(1/7 * Q_2);

% out of sample
Q1_raw = csvread('Q1_raw.csv',1,2);
Q2_raw = csvread('Q2_raw.csv',1,2);
Q3_raw = csvread('Q3_raw.csv',1,2);
Q4_raw = csvread('Q4_raw.csv',1,2);
Q_raw = {Q1_raw Q2_raw Q3_raw Q4_raw};
outsample_scenario = min([size(Q1_raw,1) size(Q2_raw,1) size(Q3_raw,1) size(Q4_raw,1)]);
% x_1_outofsample = [0.2157 0.1380 0.0579 0.1718 0.0944 0 0.1522 0.1700];
% x_2_outofsample = [0.2303 0.1518 0.0767 0.2040 0.0846 0.0072 0.0619 0.1836];
Q_1_outofsample = [];
Q_2_outofsample = [];
for scenario = 1 : outsample_scenario
    y_1_outofsample = [0 0 0 0];
    y_2_outofsample = [0 0 0 0];
    y_1_outofsample(1) = max([0, R{1}(11)-Q_raw{1}(scenario,:)*x_1']);
    y_1_outofsample(2) = max([0, R{2}(11)-Q_raw{2}(scenario,:)*x_1']);
    y_1_outofsample(3) = max([0, R{3}(11)-Q_raw{3}(scenario,:)*x_1']);
    y_1_outofsample(4) = max([0, R{4}(11)-Q_raw{4}(scenario,:)*x_1']);
    y_2_outofsample(1) = max([0, R{1}(11)-Q_raw{1}(scenario,:)*x_2']);
    y_2_outofsample(2) = max([0, R{2}(11)-Q_raw{2}(scenario,:)*x_2']);
    y_2_outofsample(3) = max([0, R{3}(11)-Q_raw{3}(scenario,:)*x_2']);
    y_2_outofsample(4) = max([0, R{4}(11)-Q_raw{4}(scenario,:)*x_2']);
    Q_1_outofsample = [Q_1_outofsample costs * y_1_outofsample'];
    Q_2_outofsample = [Q_2_outofsample costs * y_2_outofsample'];
end
z_outofsample_1 = c * x_1' + sum(1/outsample_scenario * Q_1_outofsample);
z_outofsample_2 = c * x_2' + sum(1/outsample_scenario * Q_2_outofsample);