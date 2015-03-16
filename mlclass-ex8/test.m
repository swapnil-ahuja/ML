params = [sin(1:12)';cos(1:10)'];
Y = reshape(sin(1:30), 6, 5);
R = reshape(sin(1:30), 6, 5) > 0.5;

[J, grad] = cofiCostFunc(params, Y, R, 5, 6, 2, 3)