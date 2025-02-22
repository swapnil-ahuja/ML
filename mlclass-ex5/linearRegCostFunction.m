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
%disp(size(X)),disp(X(:,2:end));
%disp(size(theta)),disp(theta(2:end,:));
%X=X(:,2:end);
%theta=theta(2:end,:);
h=(X*theta);
a1=h-y;
a2=a1.^2;
%disp(a2);
a3=theta(2:end,:).^2;
J=(1/(2*m))*sum(a2);
J=J+(lambda/(2*m))*sum(a3);
grad(1,1)=1/m.*sum((h-y).*X(1,1));
grad(2:end,[1])=(1/m.*((h-y)'*X(:,2:end)))'+((lambda/m).*theta(2:end,[1]));










% =========================================================================

grad = grad(:);

end
