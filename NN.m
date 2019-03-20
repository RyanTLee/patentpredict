%% Neural Network with data samples
clear;
clc;
close all;

%% Part 1: Split Data
% load X and Y from 1945 to 2014
P14_X = load('p14_X4.mat');
P14_Y = load('p14_Y4.mat');
m = size(P14_X.X,1);
X = zscore(P14_X.X); % do not add offset feature, normalize
y = (P14_Y.y+ones(m,1)); % No zeros!

% Use only the most recent, 500,000 examples, which have features that are
% comparable in terms of features, as features scale increase with time
size_X  = size(X,1);
m       = size_X;
X       = X(size_X-(m-1):size_X,:);
y       = y(size_X-(m-1):size_X,:);

% percentage of each label:
n_0 = sum(y==1);
n_1 = sum(y==2);
n_2 = sum(y==3);
n_total = n_0+n_1+n_2;
l_0 = n_0/n_total; 
l_1 = n_1/n_total; 
l_2 = n_2/n_total; 

figure 
histogram(y)
title('Plot of Abandoned vs. Issued for Data');
xlabel('0: Abandoned, 1: Issued, 2: pending > 2 years');
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

%% Part 2: Initialize NN for patent example set
warning('off');

% specify number of hidden layers
input_layer_size = 7; % number of features
hidden_layer_size = 10; % 
num_labels = 3;

%% Part 3: call nnCostFunction *FROM COURSERA PROGRAMMING EXCERCISE 4: Neural Networks*
% initialize weights for Neural Net
% Initialize our weights
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

options = optimset('MaxIter', 50);
lambda = 0.1;                          %  You should also try different values of lambda
                                     % Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_tr, y_tr, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%% Part 4: Use learned parameters to predict on cross-validation set
p  = predictNN( Theta1, Theta2, X_cv );
p0 = (p==1);
p1 = (p==2);
p2 = (p==3);

[F1_0, precision_0, recall_0] = prmetric (p0,y_cv, 1); 
[F1_1, precision_1, recall_1] = prmetric (p1,y_cv, 2);
[F1_2, precision_2, recall_2] = prmetric (p2,y_cv, 3);


