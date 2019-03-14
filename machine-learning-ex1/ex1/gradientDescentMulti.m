function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
GD = zeros(1,size(X,2));
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    h_theta = X * theta;% get a 47*1 matrix
    diff = h_theta - y;% 47*1
    for i = 1:size(X,2)
        GD(1,i) = 1/m*transpose(diff)*X(:,i);
        theta(i,1) = theta(i,1) - alpha * GD(1,i);
    end
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
    if iter~=1 && J_history(iter) > J_history(iter-1)
        fprint('Error!It is not decreasing!')
        return
    end
end

end
