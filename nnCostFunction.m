% GRADED FUNCTION: nnCostFunction
function [J grad] = nnCostFunction(nn_params, input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);                               % Setup some useful variables
J = 0;                                        % You need to return the following variables correctly 
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
%index an identity matrix by the y output values
eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);

% PART 1: FEED FORWARD PROPAGATION

% feature layer -> hidden layer
a1 = [ones(m,1) X];  
z2 = a1*Theta1';
a2 = sigmoid(z2);
 
% hidden layer -> output layer, sigmoid activation
a2      = [ones(size(z2,1),1) a2];
z3      = a2*Theta2';
a3      = sigmoid(z3);

% soft max activation
% z_norm  = sum(exp(a2*Theta2'),2);
% a3      = exp(a2*Theta2')./z_norm; 

% z_norm = sum(exp(z2*Theta2'),2);
% a2 = exp(z2*Theta2')./z_norm;

% cost from output layer
J = -1/m*sum(sum(y_matrix.*log(a3)+(1-y_matrix).*log(1-a3)));
reg_term = lambda/(2*m)*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));
J = J + reg_term;

% PART 2: BACK PROPAGATION
% invsigmoid(u) = sigmoid(u).*(1-sigmoid(u))
d3 = a3 - y_matrix; % mxk
Theta2_s  = Theta2(:,2:end);
d2 = (d3*Theta2_s).*sigmoid(z2).*(1-sigmoid(z2));

D1 = d2'*a1;
D2 = d3'*a2;
% Theta1_grad = Theta1_grad + D1;
% Theta2_grad = Theta2_grad + D2;

% PART 3: Regularization of gradient
Theta1_reg = Theta1; % Do NOT rewrite Theta1 or Theta2!
Theta2_reg = Theta2;
Theta1_reg(:,1) = 0;
Theta2_reg(:,1) = 0;
Theta1_grad_reg = (lambda/m)*Theta1_reg;
Theta2_grad_reg = (lambda/m)*Theta2_reg;

Theta1_grad = D1/m + Theta1_grad_reg;
Theta2_grad = D2/m + Theta2_grad_reg;


% softmax to get probability that an output belongs to the ith class for each example:
% z_norm = sum(exp(z2*Theta2'),2);
% a2 = exp(z2*Theta2')./z_norm;
% store the maximum probable class of the softmax output layer
%[Y p] = max(a3,[],2);
% ============================================================

grad = [Theta1_grad(:) ; Theta2_grad(:)];     % Unroll gradients

end