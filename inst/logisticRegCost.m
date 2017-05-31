function [J, grad] = logisticRegCost(theta, X, y, lambda)
#LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
#regularization
#   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
#   theta as the parameter for regularized logistic regression and the
#   gradient of the cost w.r.t. to the parameters. 

# Initialize some useful values
m = length(y); # number of training examples

J = 0;
grad = zeros(size(theta));


no_of_features = size(X, 2);
h_function = sigmoid(X*theta);
pos_cost = -y'*log(h_function);
neg_cost = (1 - y)'*log(1 - h_function);

non_reg_term =  (1/m)*(pos_cost-neg_cost);

theta_shifted = theta(2:end,:);

reg_term = (lambda/(2*m))*(theta_shifted'*theta_shifted);


J = non_reg_term + reg_term;

error_diff = h_function - y;


# we could vectorize this too
for i = 1:no_of_features
	if (i == 1)
		grad(i)= (1/m)*error_diff'*X(:,i);
	else
		grad(i)= (1/m)*error_diff'*X(:,i) + ((lambda/m)*theta(i));
	endif
endfor



grad = grad(:);

endfunction





%!test
%! X = [ones(3,1) magic(3)];
%! y = [1 0 1]';
%! theta = [-2 -1 1 2]';
%! [j g] = logisticRegCost(theta, X, y, 0);
%! j_expected = 4.6832;
%! g_expected = [ 0.31722 0.87232 1.64812 2.23787]';
%! assert(j, j_expected, 1);
%! assert(g, g_expected, 1);


%!test
%! X = [ones(3,1) magic(3)];
%! y = [1 0 1]';
%! theta = [-2 -1 1 2]';
%! [j g] = logisticRegCost(theta, X, y, 4);
%! j_expected = 8.6832;
%! g_expected = [ 0.31722 -0.46102 2.98146 4.90454 ]';
%! assert(j, j_expected, 1);
%! assert(g, g_expected, 1);


