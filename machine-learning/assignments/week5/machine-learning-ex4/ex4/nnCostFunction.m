function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Tutorial at
% https://www.coursera.org/learn/machine-learning/programming/AiHgN/neural-network-learning/discussions/threads/QFnrpQckEeWv5yIAC00Eog


%%%% Part 1: Forward propagation

% y is a 5000 x 1 matrix that contains the label value in each row, e.g.
% [ 10
%    8
%    2
%   ..]
% y_matrix is a 5000 x 10 matrix (10 = number of labels), with each row containing only one 1 (for the label index) and all others 0
y_matrix = eye(num_labels)(y,:); % 5000 x 10

a1 = [ones(m, 1) X]; % 5000 x 401

z2 = a1 * Theta1'; % Theta1 = 25 x 401; (5000 x 401) * (401 x 25) = 5000 x 25

a2 = sigmoid(z2); % 5000 x 25

a2_size = size(a2, 1);
a2 = [ones(a2_size, 1) a2]; % 5000 x 26

z3 = a2 * Theta2'; % Theta2 = 10 x 26; (5000 x 26) * (26 x 10) = 5000 x 10

a3 = sigmoid(z3); % 5000 x 10

% The statement below with matrix multiplication does not work.
% Refer to https://www.coursera.org/learn/machine-learning/discussions/weeks/5/threads/ag_zHUGDEeaXnBKVQldqyw
% and https://www.coursera.org/learn/machine-learning/discussions/all/threads/AzIrrO7wEeaV3gonaJwAFA
% for an explanation.
%J = (1 / m) * ( ((-1 * y_matrix') * log(a3)) - ((1 - y_matrix)' * log(1 - a3)) );

cost_matrix = ((-1 * y_matrix') * log(a3)) - ((1 - y_matrix)' * log(1 - a3));

cost_matrix = eye(size(cost_matrix)) .* cost_matrix;

J = (1 / m) * sum(sum(cost_matrix));

Theta1_regularized = Theta1;
Theta1_regularized(1) = 0;

Theta1_regularized_squared = Theta1_regularized' * Theta1_regularized;
Theta1_regularized_squared = eye(size(Theta1_regularized_squared)) .* Theta1_regularized_squared;

Theta2_regularized = Theta2;
Theta2_regularized(1) = 0;

Theta2_regularized_squared = Theta2_regularized' * Theta2_regularized;
Theta2_regularized_squared = eye(size(Theta2_regularized_squared)) .* Theta2_regularized_squared;

J = J + ( (lambda / (2 * m)) * ( sum(sum(Theta1(:,2:end) .^ 2)) + sum(sum(Theta2(:,2:end) .^ 2)) ) );


%%%% Part 2: Backward propagation (unregularized)
% https://www.coursera.org/learn/machine-learning/discussions/all/threads/a8Kce_WxEeS16yIACyoj1Q

d3 = a3 - y_matrix;     % a3 = 5000 x 10; y_matrix = 5000 x 10

d2 = (d3 * Theta2(:,2:end)) .* sigmoidGradient(z2);     % d3 = 5000 x 10; Theta2 = 10 x 26; Theta2(:,2:end) = 10 x 25; z2 = 5000 x 25

Delta1 = d2' * a1;     % d2 = 5000 x 25; a1 = 5000 x 401; Delta1 = 25 x 401

Delta2 = d3' * a2;     % d3 = 5000 x 10; a2 = 5000 x 26; Delta2 = 10 x 26

Theta1_grad = (1 / m) * Delta1;

Theta2_grad = (1 / m) * Delta2;



%%%% Part 2: Backward propagation (regularized)

Theta1(:,1) = 0;     % sets the first column in matrix to 0's

Theta2(:,1) = 0;     % sets the first column in matrix to 0's

Theta1_grad = Theta1_grad + ((lambda / m) * Theta1);

Theta2_grad = Theta2_grad + ((lambda / m) * Theta2);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
