function [ F1, precision, recall ] = prmetric( predict, labels, k )
%Get the precision, recall, and F1 score 
%   given the predictions, labels, and class k
TP = sum(predict & labels==k);
FP = sum(predict & labels~=k);
FN = sum(~predict & labels==k);

precision   = TP/(TP+FP);
recall      = TP/(TP+FN);
F1          = 2*precision*recall/(precision+recall);
end

