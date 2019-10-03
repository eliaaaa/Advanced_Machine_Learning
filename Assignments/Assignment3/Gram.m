% Calculate Gram

clear;
load data1.mat;

N = size(TrainingX, 1);

% Compute variance of kernel
squared_distance = zeros(N, N);
for i = 1:N
    for j = 1:N
        squared_distance(i, j) = norm(TrainingX(i, :) - TrainingX(j, :));
    end
end

kernel_variance = sum(sum(squared_distance)) / N^2;

% Compute the Gram Matrix

Gram_matrix = zeros(N, N);
for i = 1:N
    for j = 1:N
        Gram_matrix(i, j) = exp(-(squared_distance(i, j)) / (2 * kernel_variance));
    end
end

clear squared_distance;

save gram.mat;