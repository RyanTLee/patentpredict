%% Logistic regression with data samples
clear;
clc;
close all;

%% Part 1: Split Data
% load X and Y from 1945 to 2014
P14_X = load('p14_X4.mat');
P14_Y = load('p14_Y4.mat');
m = size(P14_X.X,1);
X = [ones(m,1) zscore(P14_X.X)]; % add offset feature, normalize
y = P14_Y.y;

% Use only the most recent, 500,000 examples, which have features that are
% comparable in terms of features, as features scale increase with time
size_X  = size(X,1);
m       = size_X;
X       = X(size_X-(m-1):size_X,:);
y       = y(size_X-(m-1):size_X,:);

% percentage of each label:
n_0 = sum(y==0);
n_1 = sum(y==1);
n_2 = sum(y==2);
n_total = n_0+n_1+n_2;
l_0 = n_0/n_total; 
l_1 = n_1/n_total; 
l_2 = n_2/n_total; 

figure 
set(gcf,'color','w');
histogram(y)
title('Plot of Data Labels');
xlabel('0: Abandoned, 1: Issued, 2: pending > 32mo.');
ylabel('Number of Patents, Data');

% Divide most recent examples into a training, cross-val and test set
n_training = 0.6*m;
n_cv       = 0.2*m;
n_test     = 0.2*m;

% Index 
X_tr = X(1:n_training,:);
y_tr = y(1:n_training,:);
m_tr = size(X_tr, 1);

X_cv = X(n_training+1:(n_training+n_cv),:);
y_cv = y(n_training+1:(n_training+n_cv),:);
m_cv  = size(X_cv, 1);

X_text = X((n_training+n_cv)+1:m,:);
y_test = y((n_training+n_cv)+1:m,:);
m_test  = size(X_text, 1);

%% Part 2: Train Data with fminunc, regularized logistic regression cost fx
y0_tr = (y_tr == 0);
y1_tr = (y_tr == 1);
y2_tr = (y_tr == 2);

initial_theta = zeros(size(X_tr, 2), 1);     %[n+1 x 1]                   % Initialize fitting parameters
lambda        = 0.5;                                            % Set regularization parameter lambda to 1 (you should vary this)
options       = optimset('GradObj', 'on', 'MaxIter', 400);      % Set Options

[theta0, J0, exit_flag0] = fminunc(@(t)(costFunctionLogReg(X_tr, y0_tr,t, lambda)), initial_theta, options);
[theta1, J1, exit_flag1] = fminunc(@(t)(costFunctionLogReg(X_tr, y1_tr,t, lambda)), initial_theta, options);
[theta2, J2, exit_flag2] = fminunc(@(t)(costFunctionLogReg(X_tr, y2_tr,t, lambda)), initial_theta, options);

% predictions on cross-validation set
p0_p = predictprob(theta0,X_cv);
p1_p = predictprob(theta1,X_cv);
p2_p = predictprob(theta2,X_cv);

% predictions of classes on cross-validation set
log_threshold = 0.5;
p0 = predict(theta0,X_cv, log_threshold);
p1 = predict(theta1,X_cv, log_threshold);
p2 = predict(theta2,X_cv, log_threshold);

%% Compute Reciever Operating Characteristic Curve for classifier output
[Xpc0, Ypc0, Tpc0, AUC0] = perfcurve(y_cv,p0_p,0);
[Xpc1, Ypc1, Tpc1, AUC1] = perfcurve(y_cv,p1_p,1);
[Xpc2, Ypc2, Tpc2, AUC2] = perfcurve(y_cv,p2_p,2);

% random classifier for comparison
Xrand = linspace(0,1,5000);
Yrand = Xrand;

figure 
set(gcf,'color','w');
plot(Xpc0,Ypc0);
hold on;
plot(Xpc1,Ypc1);
plot(Xpc2,Ypc2);
plot(Xrand, Yrand);
legend('Abandoned','Issued','Pending more than 32 months','Random Classifier');
xlabel('False positive rate');
ylabel('True positive rate');
title('ROC Curves: Logistic Regression with 3 classes, Lambda = 0.5');
hold off

[F1_0, precision_0, recall_0] = prmetric (p0,y_cv, 0); 
[F1_1, precision_1, recall_1] = prmetric (p1,y_cv, 1);
[F1_2, precision_2, recall_2] = prmetric (p2,y_cv, 2);
