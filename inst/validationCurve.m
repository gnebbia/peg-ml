function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
#VALIDATIONCURVE Generate the train and validation errors needed to
#plot a validation curve that we can use to select lambda
#   [lambda_vec, error_train, error_val] = ...
#       VALIDATIONCURVE(X, y, Xval, yval) returns the train
#       and validation errors (in error_train, error_val)
#       for different values of lambda. You are given the training set (X,
#       y) and validation set (Xval, yval).
#

lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);


for i = 1:length(lambda_vec)
  lambda = lambda_vec(i);
  theta = trainLinearReg(X, y, lambda);

  # remember that here we are computing the errors, so in order to compute
  # errors lambda should be zero, since we are just reusing the cost function
  # to compute errors, in order to avoid rewriting code
  [J_train, grad] = linearRegCost(X, y, theta, 0);
  [J_cv , grad] = linearRegCost(Xval, yval, theta, 0);
  error_train(i) = J_train;
  error_val(i) = J_cv;
endfor


endfunction


%!test
%! X = [1 2 ; 1 3 ; 1 4 ; 1 5];
%! y = [7 6 5 4]';
%! Xval = [1 7 ; 1 -2];
%! yval = [2 12]';
%! [lambda_vec, error_train, error_val] = validationCurve(X,y,Xval,yval );
%! lambda_vec_expected = [ 0.00000 0.00100 0.00300 0.01000 0.03000 0.10000 ...
%! 0.30000 1.00000 3.00000 10.00000 ]';
%! error_train_expected = [ 0.00000 0.00000 0.00000 0.00000 0.00002 0.00024 ...
%! 0.00200 0.01736 0.08789 0.27778 ]';
%! error_val_expected = [ 0.25000 0.25055 0.25165 0.25553 0.26678 0.30801 ... 
%! 0.43970 1.00347 2.77539 6.80556]';

%! assert(lambda_vec, lambda_vec_expected, 1);
%! assert(error_val, error_val_expected, 1);
%! assert(error_train, error_train_expected, 1);
