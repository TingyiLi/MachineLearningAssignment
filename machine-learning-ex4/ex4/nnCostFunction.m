function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
a_1 = [ones(size(X,1),1) X];%add bias unit for input layer 5000*401
z_2 = a_1*transpose(Theta1);%401*25 5000*25
a_2 = sigmoid(z_2);
a_2 = [ones(size(a_2,1),1) a_2];%5000*26
z_3 = a_2*transpose(Theta2);%10*26 5000*10
a_3 = sigmoid(z_3);%5000*10
y_out = zeros(m,num_labels);
for i = 1:size(y_out,1)
    y_out(i,y(i,1)) = 1;%5000*10
end
J = sum(sum(-y_out.*log(a_3)-(1-y_out).*log(1-a_3)))/m;%5000*10
reg = lambda/(2*m)*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));
J = J+reg;

% Theta1_grad = zeros(size(Theta1)); 25*401
% Theta2_grad = zeros(size(Theta2)); 10*26
delta1 = zeros(size(Theta1));%25*401
delta2 = zeros(size(Theta2));%10*26
for i = 1:m
    a_1 = X(i,:);%1*400
    a_1 = [1 a_1];%add bias unit for input layer 1*401
    z_2 = a_1*transpose(Theta1);%1*25 
    a_2 = sigmoid(z_2);
    a_2 = [1 a_2];%1*26
    z_3 = a_2*transpose(Theta2);%10*26 1*10
    a_3 = sigmoid(z_3);%1*10
    e_3 = a_3 - y_out(i,:);%1*10
    e_2 = e_3*Theta2(:,2:end).*sigmoidGradient(z_2);%1*25
    delta2 = delta2 + transpose(e_3)*a_2;
    delta1 = delta1 + transpose(e_2)*a_1;
end
theta_tmp1 = Theta1;
theta_tmp2 = Theta2;
theta_tmp1(:, 1)=0;
theta_tmp2(:, 1)=0;
Theta1_grad = delta1/m + lambda/m*theta_tmp1;
Theta2_grad = delta2/m + lambda/m*theta_tmp2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
