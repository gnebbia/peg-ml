#SIGMOID Compute sigmoid functoon
#   J = SIGMOID(z) computes the sigmoid of z.
function g = sigmoid(z)

g = 1.0 ./ (1.0 + exp(-z));

end



%!test
%! s = sigmoid(1200000);
%! s_expected = 1;
%! assert(s, s_expected, 1);

%!test
%! s = sigmoid(-25000);
%! s_expected = 0;
%! assert(s, s_expected, 1);

%!test
%! s = sigmoid(0);
%! s_expected = 0.5;
%! assert(s, s_expected, 1);

%!test
%! s = sigmoid([4 5 6]);
%! s_expected = [ 0.9820 0.9933 0.9975 ];
%! assert(s, s_expected, 1);
