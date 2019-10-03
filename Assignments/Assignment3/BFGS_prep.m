clear;
load data1.mat;

index1 = randperm(5000, 2000);
temp = 5001:10000;
index2 = temp(randperm(5000, 2000));

TrainingX = cat(1, TrainingX(index1, :), TrainingX(index2, :));
TrainingY = cat(1, TrainingY(index1, :), TrainingY(index2, :));

N = size(TrainingX, 1);

Norm = zeros(N, N);
for i = 1:N
    for j = 1:N
        Norm(i, j) = norm(TrainingX(i, :) - TrainingX(j, :));
    end
end

delta2 = sum(sum(Norm)) / N^2;

gram = zeros(N, N);
for i = 1:N
    for j = 1:N
        gram(i, j) = exp(-Norm(i, j) / (2 * delta2));
    end
end

clear squared_distance;

save bfgs_prep.mat;