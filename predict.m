function p = predict(theta, X,log_threshold)

m = size(X, 1);                     % Number of training examples
p = zeros(m, 1);                    % Return the following variable correctly

% ====================== YOUR CODE HERE ======================
p = sigmoid(X*theta)>= log_threshold;

% =============================================================

end