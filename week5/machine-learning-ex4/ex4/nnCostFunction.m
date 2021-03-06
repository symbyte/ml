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

XwithBias = [ones(m, 1), X];

z2 = (Theta1 * XwithBias');

a2 = sigmoid(z2);

a3 = sigmoid(Theta2 * [ones(1, m); a2]);

for i = 1:m
  yVec = createYLogicalVector(num_labels, y, i);
  a3i = a3(:,i);
  a2i = a2(:,i);
  z2i = z2(:,i);
  a1i = XwithBias(i,:);
  J += sum(-yVec' * log(a3i) - (1 - yVec') * log(1 - a3i));

  % run back prop
  d3i = a3i - yVec;
  d2i = (Theta2' * d3i) .* sigmoidGradient([1; z2i]);

  Theta1_grad = Theta1_grad + (d2i * a1i)(2:end,:);
  Theta2_grad = Theta2_grad + (d3i * [1; a2i]');
endfor

J /= m;
Theta1_grad /= m;
Theta2_grad /= m;

squaredTheta1 = Theta1 .^ 2;
squaredTheta2 = Theta2 .^ 2;
squaredTheta1WithoutBias = squaredTheta1(:,2:end);
squaredTheta2WithoutBias = squaredTheta2(:,2:end);

regularizationTerm = (lambda/(2 * m)) * ...
  (sum(squaredTheta1WithoutBias(:)) + sum(squaredTheta2WithoutBias(:)));
J += regularizationTerm;

non_bias_Theta1 = [zeros(size(Theta1,1), 1) Theta1(:,2:end)];
non_bias_Theta2 = [zeros(size(Theta2,1), 1) Theta2(:,2:end)];
Theta1_reg_term = (lambda/m) * non_bias_Theta1;
Theta2_reg_term = (lambda/m) * non_bias_Theta2;
Theta1_grad += Theta1_reg_term;
Theta2_grad += Theta2_reg_term;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

function yVec = createYLogicalVector(number_of_labels, y, example_number)
  yVec = zeros(number_of_labels, 1);
  yVec(y(example_number)) = 1;
end
