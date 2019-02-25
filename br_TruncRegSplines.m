function [X, x_mu] = br_TruncRegSplines(x, k, q)

% Demean the predictors
n = length(x);
x_mu = mean(x);
x = x - mean(x);

% Find the knots
V = 100/k;
c = prctile(x, linspace(V, (100-V), k-1));

% Form the X matrix
X = zeros(n, 3 + k - 1);
X(:,1:3) = [x, x.^2, x.^3];
X(:,4:end) = max(bsxfun(@minus, x, c),0).^q;

end
