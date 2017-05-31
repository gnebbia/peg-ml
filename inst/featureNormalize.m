function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

mu = mean(X);
X_norm = bsxfun(@minus, X, mu);

sigma = std(X_norm);
X_norm = bsxfun(@rdivide, X_norm, sigma);

endfunction


%!test
%! [Xn mu sigma] = featureNormalize([1 ; 2 ; 3]);
%! Xn_expected = [ -1 0 1 ]';
%! mu_expected = 2;
%! sigma_expected = 1;
%! assert(Xn, Xn_expected, 1);
%! assert(mu, mu_expected, 1);
%! assert(sigma, sigma_expected, 1);


%!test
%! [Xn mu sigma] = featureNormalize(magic(3));
%! Xn_expected = [ 1.13389 -1.00000 0.37796; ...
%! -0.75593 0.00000 0.75593; -0.37796 1.00000 -1.13389 ];
%! mu_expected = [ 5 5 5 ];
%! sigma_expected = [ 2.6458 4.0000 2.6458 ];
%! assert(Xn, Xn_expected, 1);
%! assert(mu, mu_expected, 1);
%! assert(sigma, sigma_expected, 1);


%!test
%! [Xn mu sigma] = featureNormalize([-ones(1,3); magic(3)]);
%! Xn_expected = [  -1.21725  -1.01472  -1.21725; ...
%! 1.21725  -0.56373   0.67625; ...
%! -0.13525   0.33824   0.94675; ...
%! 0.13525   1.24022  -0.40575 ];
%! mu_expected = [ 3.5000 3.5000 3.5000 ];
%! sigma_expected = [ 3.6968 4.4347 3.6968 ];
%! assert(Xn, Xn_expected, 1);
%! assert(mu, mu_expected, 1);
%! assert(sigma, sigma_expected, 1);
