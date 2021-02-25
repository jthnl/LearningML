% ===== I. PREPROCESS DATA ============================================== %

TBL_FN = "COVID-19_formatted_dataset.csv";
TBL_raw = readtable(TBL_FN);

% data constants
TBL_raw_cvdT = 598;                             % num total tests
TBL_raw_cvdP = 81;                              % num positive tests
TBL_raw_cvdN = TBL_raw_cvdT - TBL_raw_cvdP;     % num negative tests

TBL_data_cvdN = zeros(TBL_raw_cvdN, 16);
TBL_data_cvdP = zeros(TBL_raw_cvdP, 16);
TBL_data_cvdN_idx = 1;
TBL_data_cvdP_idx = 1;

for i = 1:size(TBL_raw)
    row = TBL_raw(i,:);
    % Seperate into positive from negative
    if strcmp(row{1,3}, "positive") == 1
        TBL_data_cvdP(TBL_data_cvdP_idx,1) = 2 * (row{1,2} / 19) - 1;   % normalize age
        TBL_data_cvdP(TBL_data_cvdP_idx,2) = 0.76;                      % set as 0.76
        TBL_data_cvdP(TBL_data_cvdP_idx,3:16) = row{1,4:17};
        TBL_data_cvdP_idx = TBL_data_cvdP_idx + 1;
    else
        TBL_data_cvdN(TBL_data_cvdN_idx,1) = 2 * (row{1,2} / 19) - 1;   % normalize age
        TBL_data_cvdN(TBL_data_cvdN_idx,2) = -0.76;                     % set as -0.76
        TBL_data_cvdN(TBL_data_cvdN_idx,3:16) = row{1,4:17};
        TBL_data_cvdN_idx = TBL_data_cvdN_idx + 1;
    end
end

% seperate into training (90%) and testing data (10%)
TBL_test_ceil_cvdN = ceil(TBL_raw_cvdN/10);

TBL_randperm_cvdN = randperm(size(TBL_data_cvdN,1));
TBL_test_cvdN = TBL_data_cvdN(TBL_randperm_cvdN(1:TBL_test_ceil_cvdN),:);
TBL_train_cvdN = TBL_data_cvdN(TBL_randperm_cvdN(TBL_test_ceil_cvdN+1:end),:);

TBL_test_ceil_cvdP = ceil(TBL_raw_cvdP/10);

TBL_randperm_cvdP = randperm(size(TBL_data_cvdP,1));
TBL_test_cvdP = TBL_data_cvdP(TBL_randperm_cvdP(1:TBL_test_ceil_cvdP),:);
TBL_train_cvdP = TBL_data_cvdP(TBL_randperm_cvdP(TBL_test_ceil_cvdP+1:end),:);

% combine data
TBL_training = [TBL_train_cvdP; repmat(TBL_train_cvdP,6,1); TBL_train_cvdN];
TBL_testing = [TBL_test_cvdP; TBL_test_cvdN];

% shuffle rows
TBL_training = TBL_training(randperm(size(TBL_training, 1)), :);
TBL_testing = TBL_testing(randperm(size(TBL_testing, 1)), :);

% clear all unnecessary variables
clearvars -except TBL_training TBL_testing;

% ===== II. TRAINING DATA =============================================== %

% set training data to input and output
P = TBL_training(:, [1 3:16]);
T = TBL_training(:, 2);

% nodes for each layer
n1 = 10;
n2 = 1;

% weights and biases
W1 = randn(n1, 15);
W2 = randn(n2, n1);
b1 = rand(n1,1);
b2 = rand();

% alpha and error threshold
ap = 0.01;     
et = 0.00002;

% iterations and error calculations
maxIter = 1000;
erArr = zeros(1, maxIter);
erIdx = 1;
err = [];

% train network
while erIdx < maxIter
    for idx = 1:length(P)   
        % 1. propagate input forward
        inputs = P(idx,:)';
        a1 = tansig(W1 * inputs + b1);
        a2 = tansig(W2 * a1 + b2);
        err(idx) = T(idx,:) - a2;

        % 2. backpropagate sensitivities
        tmp = [1-(a1(1)^2), 1-(a1(2)^2), 1-(a1(3)^2), 1-(a1(4)^2), 1-(a1(5)^2), 1-(a1(6)^2), 1-(a1(7)^2), 1-(a1(8)^2), 1-(a1(9)^2), 1-(a1(10)^2)];
        % == 20 nodes ==   %tmp = [1-(a1(1)^2), 1-(a1(2)^2), 1-(a1(3)^2), 1-(a1(4)^2), 1-(a1(5)^2), 1-(a1(6)^2), 1-(a1(7)^2), 1-(a1(8)^2), 1-(a1(9)^2), 1-(a1(10)^2), 1-(a1(11)^2), 1-(a1(12)^2), 1-(a1(13)^2), 1-(a1(14)^2), 1-(a1(15)^2),  1-(a1(16)^2),  1-(a1(17)^2),  1-(a1(18)^2),  1-(a1(19)^2),  1-(a1(20)^2)];
        % == 6  nodes ==   %tmp = [1-(a1(1)^2), 1-(a1(2)^2), 1-(a1(3)^2), 1-(a1(4)^2), 1-(a1(5)^2), 1-(a1(6)^2)];
        F1 = diag(tmp);
        F2 = 1-(a2^2);
        s2 = -2 * F2 * err(idx);
        s1 = F1 * W2' * s2;

        % 3. update weight and bias
        W1 = W1 - ap * s1 * inputs';
        b1 = b1 - ap * s1;
        W2 = W2 - ap * s2 * a1';
        b2 = b2 - ap * s2;
    end
    
    % 4. calculate MSE and check error threshold
    erArr(erIdx) = mse(err);
    if (erArr(erIdx) <= et) || (erIdx >= maxIter)
        break;
    end
    erIdx = erIdx + 1 % print out iteration (i.e. lazy progress bar)
end

% ===== III. TESTING DATA =============================================== %

% set testing data
P_test = TBL_testing(:, [1 3:16]);
T_test = TBL_testing(:, 2);

% calculate test data results
calc = zeros(1, length(P_test));
for k = 1:length(P_test)
    inputs = P_test(k,:)';
    a1 = tansig(W1 * inputs + b1);
    a2 = tansig(W2 * a1 + b2);
    calc(1,k) = a2;
end
resTable = [T_test(:,1), calc'];
p = 0;

% calculate confusion table
Z_cor_p = 0;
Z_inc_p = 0;
Z_cor_n = 0;
Z_inc_n = 0;

for i = 1:length(resTable)
    if (resTable(i,1) == 0.76)
        if (resTable(i,2) > 0)
            Z_cor_p = Z_cor_p + 1;  %topleft
        else
            Z_inc_p = Z_inc_p + 1;  %btmleft
        end
    else
        if (resTable(i,2) < 0)
            Z_cor_n = Z_cor_n + 1;  %topright
        else
            Z_inc_n = Z_inc_n + 1;  %btmright
        end
    end
end

Z_total = (Z_cor_p + Z_cor_n) / (Z_cor_p + Z_inc_p + Z_cor_n + Z_inc_n);

% plot graphs
figure, plot(erArr(1:erIdx));
hold on;

figure, scatter(1:61, calc);
hold on;