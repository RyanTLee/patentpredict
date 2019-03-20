function [ g ] = sigmoid( z )
%SIGMOID hypothesis of logistic regression
g = 1./(1+exp(-z));

end

