function [ p ] = predictNN( Theta1, Theta2, X )
%PREDICTNN Summary of this function goes here
%   Detailed explanation goes here
m = size(X, 1);                                % Useful values
num_labels = size(Theta2, 1);
p = zeros(size(X, 1), 1);                      % Return the following variable correctly 

% ====================== YOUR CODE HERE ======================
% I'm missing how the classifiers 1 through num layers

% PART 1: FEED FORWARD PROPAGATION

% feature layer -> hidden layer
a1 = [ones(m,1) X];  
z2 = a1*Theta1';
a2 = sigmoid(z2);
 
% hidden layer -> output layer
a2      = [ones(size(z2,1),1) a2];
z3      = a2*Theta2';
a3      = sigmoid(z3);

% soft max activation
z_norm  = sum(exp(a2*Theta2'),2);
a3      = exp(a2*Theta2')./z_norm; 

[Y p] = max(a3,[],2);

% ============================================================

end

