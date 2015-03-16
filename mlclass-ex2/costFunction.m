function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
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
%disp("size of theta="),disp(size(theta)),disp(theta);
%disp("size of X="),disp(size(X));
%disp("size of y="),disp(size(y)),disp(y);


h=sigmoid(X*theta);
%disp("value of h="),disp(h);
a1=-y.*log(h);
a2=(1.-y).*log(1.-h);
%disp("size of a2="),disp(size(a2));
%disp("size of a1="),disp(size(a1));

J=1/m*sum(a1-a2);
grad=1/m.*((h-y)'*X);






% =============================================================

end
