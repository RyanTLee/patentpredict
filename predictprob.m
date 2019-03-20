function p = predictprob(theta, X)

m = size(X, 1);                     % Number of training examples
p = zeros(m, 1);                    % Return the following variable correctly

% ====================== YOUR CODE HERE ======================
log_threshold = 0.5;
p = sigmoid(X*theta); %>= log_threshold;

% =============================================================

end