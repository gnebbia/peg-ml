function [J, grad] = linearRegCost(X, y, theta, lambda)
#LINEARREGCOST Compute cost and gradient for regularized linear 
#regression with multiple variables
#   [J, grad] = LINEARREG(X, y, theta, lambda) computes the 
#   cost of using theta as the parameter for linear regression to fit the 
#   data points in X and y. Returns the cost in J and the gradient in grad

# Initialize some useful values
m = length(y); # number of training examples

J = 0;
grad = zeros(size(theta));


h = X*theta;

theta_shifted = theta(2:end,:);
error_term = (1/(2*m))*((h-y)'*(h-y));
reg_term = (lambda/(2*m)) * (theta_shifted'*theta_shifted);

J = error_term + reg_term;


reg_term_grad = (lambda/m) * (theta_shifted);

reg_term_grad = [0; reg_term_grad];

grad = (1/m)* (X'*(h-y)) + reg_term_grad;


grad = grad(:);

endfunction


%!test
%! X = [[1 1 1]' magic(3)];
%! y = [7 6 5]';
%! theta = [0.1 0.2 0.3 0.4]';
%! [J g] = linearRegCost(X, y, theta, 0);
%! J_expected = 1.3533;
%! g_expected = [ -1.4000 -8.7333 -4.3333 -7.9333 ]';
%! assert(J, J_expected, 1);
%! assert(g, g_expected, 1);


%!test
%! X = [[1 1 1]' magic(3)];
%! y = [7 6 5]';
%! theta = [0.1 0.2 0.3 0.4]';
%! [J g] = linearRegCost(X, y, theta, 7);
%! J_expected = 1.6917;
%! g_expected = [ -1.4000 -8.2667 -3.6333 -7.0000 ]';
%! assert(J, J_expected, 1);
%! assert(g, g_expected, 1);



%!test
%! X = [1 2 3 4];
%! y = 5;
%! theta = [0.1 0.2 0.3 0.4]';
%! [J g] = linearRegCost(X, y, theta, 7);
%! J_expected = 3.0150;
%! g_expected = [ -2.0000 -2.6000 -3.9000 -5.2000 ]';
%! assert(J, J_expected, 1);
%! assert(g, g_expected, 1);

