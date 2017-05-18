function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.





% =========================================================================

% According to the lectures, we could calculate this using theta' * X
% But in this case, theta is 2 x 1 (transpose would become 1 x 2) and X is 97 x 2, so elements do not match for matrix multiplication
%HX = theta' * X;

% Reversing the order gives us what we need
%HX = X * theta;

% This is the for loop solution
%for i = 1:m
%  J += ((HX(i) - y(i)) ^ 2);
%endfor

%J = J / (2 * m);

% Vectorized solution
J = (((X * theta) - y)' * ((X * theta) - y)) / (2 * m);

end
