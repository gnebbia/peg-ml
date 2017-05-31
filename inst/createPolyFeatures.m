function [X_poly] =  createPolyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);

for i = 1:p
  X_poly(:, i) = X(:,1).^(i);
endfor





% =========================================================================

# How to create polynomial features, we start from X, which is data without ones
# X_poly = polyFeatures(X, p);
# [X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
# X_poly = [ones(m, 1), X_poly];                   % Add Ones

endfunction


%!test
%! p = createPolyFeatures([1:3]',4);
%! p_expected = [1    1    1    1; 2    4    8   16; 3    9   27   81];
%! assert(p, p_expected, 1);
