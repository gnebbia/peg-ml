function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
#LEARNINGCURVE Generates the train and cross validation set errors needed 
#to plot a learning curve
#   [error_train, error_val] = ...
#       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
#       cross validation set errors for a learning curve. In particular, 
#       it returns two vectors of the same length - error_train and 
#       error_val. Then, error_train(i) contains the training error for
#       i examples (and similarly for error_val(i)).
#
#   In this function, you will compute the train and test errors for
#   dataset sizes from 1 up to m. In practice, when working with larger
#   datasets, you might want to do this in larger intervals.
#

# Number of training examples
m = size(X, 1);

error_train = zeros(m, 1);
error_val   = zeros(m, 1);


for i = 1:m
  subset_X = X(1:i, :);
  subset_y = y(1:i, :);
  theta = trainLinearReg(subset_X, subset_y, lambda);

  # notice that the error must be computed without any regularization term
  # so we pass a 0 instead of lambda when computing the cost
  [J_train, grad] = linearRegCost(subset_X, subset_y, theta, 0);
  error_train(i) = J_train;
  [J_cv, grad] = linearRegCost(Xval, yval, theta, 0);
  error_val(i) = J_cv;
endfor

# -------------------------------------------------------------

endfunction

