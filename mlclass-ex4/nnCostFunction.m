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
a1=X;
a1(:,size(a1,2)+1)=0;
k=size(a1,2);
a=k;
for i = 1:k-1
a1(:,a)=a1(:,a-1);
a=a-1;
endfor

 a1(:,1)=1;
 
%disp("size of a1"),disp(size(a1));
z2=a1*Theta1';
a2=sigmoid(z2);
%disp("size of a2"),disp(size(a2)),disp(a2);
 a2(:,size(a2,2)+1)=0;
k=size(a2,2);
a=k;
for i = 1:k-1
a2(:,a)=a2(:,a-1);
a=a-1;
endfor
a2(:,1)=1;
 %disp("new size of a2"),disp(size(a2));
z3=a2*Theta2';
a3=sigmoid(z3);
h=a3;
%disp("size of h1"),disp(size(h));
y1=eye(num_labels)(y,:);
%disp("size of y1"),disp(size(y1));
k=size(h,2);
a4(m,1)=0;
a5(m,1)=0;
%disp("size of a4"),disp(size(a4));
%disp("size of a5"),disp(size(a5));
for a=1:k
a4=a4+(-y1(:,(a)).*log(h(:,a)));
a5=a5+(1.-y1(:,(a))).*log(1.-h(:,a));
endfor
%disp("size of a4"),disp(size(a4));
%disp("size of a5"),disp(size(a5));
J1=((1/m)*sum(a4-a5));
%disp("size of Theta1"),disp(size(Theta1));
%disp("size of Theta2"),disp(size(Theta2));
r1=sum(Theta1(:,2:end).^2);
r2=sum(Theta2(:,2:end).^2);
r=sum(r1)+sum(r2);
%disp("lambda="),disp(lambda);
f=lambda/(2*m);
%disp("f="),disp(f);
J=J1+f*r;
y1=eye(num_labels)(y,:);
%del1(a,:)=0;
%del2(a,:)=0;

d3=a3-y1;
d2=(d3*Theta2(:,2:end)).*sigmoidGradient(z2);
del1=d2'*a1;
del2=d3'*a2;



Theta1_grad=(1/m)*del1;
Theta2_grad=(1/m)*del2;
reg1=(lambda/m).*Theta1(:,2:end);
reg2=(lambda/m).*Theta2(:,2:end);

Theta1_grad(:,2:end)=Theta1_grad(:,2:end)+reg1;
Theta2_grad(:,2:end)=Theta2_grad(:,2:end)+reg2;














% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
