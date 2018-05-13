function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


hypothesis = X * theta;
errors = hypothesis - y;
squaredErrors = errors .^ 2;

thetaWithoutBias = theta(2:end);
regularizationTerm = (lambda / (2 * m)) * sum(thetaWithoutBias .^ 2);

J = (1/(2 * m) * sum(squaredErrors)) + regularizationTerm;

regularizationGradient = (lambda/m) * theta;
regularizationGradient(1) = 0;

for j = 1:length(grad')
  gradient = (( 1 / m ) * sum(errors' * X(:,j)));
  grad(j) =  gradient + regularizationGradient(j);
endfor










% =========================================================================

grad = grad(:);

end
