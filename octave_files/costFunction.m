function [J, grad] = costFunction(theta, X, y, lambda)
%COSTFUNCTION Compute cost and gradient for regularized logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

hX = sigmoid(X * theta);

%For values between 0.99999 and 1 octave sees as 1 :(
hX = hX - ((hX == 1) * 0.00001);

%basic cost w/o regularized term
J = -(1/m) * (y' * log(hX) + (1 - y)' * log(1 - hX));

%adding regularized term
J = J + (lambda / (2*m)) * [0 theta(2:end)'] * theta; 

%gradient (with regularized term)
grad = (1/m) * X' * (hX - y) + (lambda / m) * [0 ; theta(2:end)];


% =============================================================

end
