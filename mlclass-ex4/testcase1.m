disp("displaying result");
[J grad] = nnCostFunction(sec(1:1:32)', 2, 4, 4, reshape(tan(1:32), 16, 2) / 5, 1 + mod(1:16,4)', 0.1)
disp("J="),disp(J);
disp("grad"),disp(size(grad)),disp(grad);