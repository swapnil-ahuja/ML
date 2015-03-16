[lambda_vec error_train error_val] = validationCurve([ones(10,1) sec(1:1.5:15)' tan(1:1.5:15)'], cot(1:3:30)', [1 23.5 12.4; 1 64.3 10.1; 1 76.4 9.8; 1 34.2 15.2; 1 59.5 13.5], [13;24;53;34;23]);
disp("lambda_vec"),disp(lambda_vec);
disp("error_train"),disp(error_train);
disp("error_val"),disp(error_val);
