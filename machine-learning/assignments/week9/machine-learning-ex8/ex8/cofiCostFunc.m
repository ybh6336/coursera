function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);


% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the
%                     partial derivatives w.r.t. to each element of Theta
%

% unregularized J
X_Theta = X * Theta';    % nm x nu
Diff = X_Theta - Y;    % nm x nu
Diff_R = Diff .* R;    % nm x nu
Diff_R_squared = Diff_R .^ 2;    % nm x nu
J = (1 / 2) * sum(Diff_R_squared(:));    % scalar value

% unregularized gradients
X_grad = Diff_R * Theta;    % [nm x nu] * [nu x 100] = [nm x 100]
Theta_grad = Diff_R' * X;    % [nu x nm] * [nm x 100] = [nu x 100]

% regularized J
Theta_squared = Theta .^ 2;
Theta_reg = (lambda / 2) * sum(Theta_squared(:));
X_squared = X .^ 2;
X_reg = (lambda / 2) * sum(X_squared(:));
J = J + Theta_reg + X_reg;

% regularized gradients
X_grad = X_grad + (lambda * X);
Theta_grad = Theta_grad + (lambda * Theta);

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
