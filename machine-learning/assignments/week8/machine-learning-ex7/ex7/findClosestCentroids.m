function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

%for i = 1:size(X,1)
%    min_dist_sum = realmax;
%    min_dist_idx = intmax;
%    for c = 1:size(centroids,1)
%        % find distance between X(i) and centroids(c)
%        dist = minus(X(i, :), centroids(c, :));
%        temp_dist_sum = sum(dist.^2,2);
%
%        if temp_dist_sum < min_dist_sum
%            min_dist_sum = temp_dist_sum;
%            min_dist_idx = c;
%        end
%
%        idx(i) = min_dist_idx;
%    end
%end


% The above code iterates over the examples to find distance of each training example
% from each centroid. The code below uses matrix arithmetic to achieve the same thing.
distance = zeros(size(X,1), size(centroids,1));

for i = 1:size(centroids,1)
    D = bsxfun(@minus, X, centroids(i, :));
    S = sum(D.^2, 2);
    distance(:, i) = S;
end

[minimum, index] = min(distance, [], 2);

idx = index;

% =============================================================

end
