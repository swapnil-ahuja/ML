function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)
%Theta1 = reshape(sin(0 : 0.5 : 5.9), 4, 3);
%Theta2 = reshape(sin(0 : 0.3 : 5.9), 4, 5);
%X = [1 -1 ; 4 1.5 ; 3.5 2.8 ; 1 1]
% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.

%disp("size of Theta1"),disp(size(Theta1)),disp(Theta1);
%disp("size of Theta2"),disp(size(Theta2)),disp(Theta2);
%disp("size of X="),disp(size(X));
 a1=X;
a1(:,size(a1,2)+1)=0;
k=size(a1,2);
a=k;
for i = 1:k-1
a1(:,a)=a1(:,a-1);
a=a-1;
endfor

 a1(:,1)=1;
 
%disp("size of a1"),disp(size(a1)),disp(a1);
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
 %disp("new size of a2"),disp(size(a2)),disp(a2);
z3=a2*Theta2';
a3=sigmoid(z3);
 %disp("size of a3"),disp(size(a3)),disp(a3);
 [b,p(:,[1])]=max(a3,[],2);

%disp("size of p"),disp(size(p));






% =========================================================================


end
