function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
%C =  0.010000;
%sigma = 1;
C1 = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma1 = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
%C1 = [0.01; 1; 3];
%sigma1 = [0.01; 1; 3];

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%for i=1:8
%for j=1:8
model= svmTrain(X, y, C1(5), @(x1, x2) gaussianKernel(x1, x2, sigma1(3)));
predictions =svmPredict(model,Xval);
%error= mean(double(predictions ~= yval));

%disp("error="),disp(error(i,j));
%endfor
%endfor
%[x,ix]=min(error(:));
%disp("x="),disp(x);
%disp("ix="),disp(ix);
%disp(error);

C=C1(5);
sigma=sigma1(3);


% =========================================================================

end
