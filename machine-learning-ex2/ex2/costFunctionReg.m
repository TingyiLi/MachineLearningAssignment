function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
h_theta = sigmoid(X*theta); 
tmp = -transpose(y)*log(h_theta)-transpose(1-y)*log(1-h_theta);
J = tmp/m+lambda/(2*m)*sum(theta(2:end,1).^2);
temp = h_theta - y;%100*1
grad(1,1) = (transpose(X(:,1))*temp)/m;
for i = 2:size(theta,1)
    temp = h_theta - y;%100*1
    grad(i,1) = ((transpose(X(:,i))*temp)+lambda*theta(i,1))/m;
end

% =============================================================

end
