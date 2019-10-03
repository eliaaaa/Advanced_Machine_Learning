clear;
load gram.mat;

% Stochastic Gradient Descent p=1
N = size(TrainingX, 1);

step_size = 3e-4; 
epsilon = 1e-5; 
lambda = 1e-3;
w = zeros(N,1); 
cost_by_iteration_sgd1 = [];
time_by_iteration_sgd1 = [];

tic;
best_w = w;
min_cost = 1000;
p = 1; 

i = 1;


while(i<20000)
    % compute the cost and gradient of the current w
    
    loss_now = loss_function(TrainingY, Gram_matrix, w, lambda);
    sample = randsample(N, p);
    gradient = loss_gradient(TrainingY(sample), Gram_matrix(sample,:), ...
        w, lambda);
    

    cost_by_iteration_sgd1(i) = loss_now;
    time_by_iteration_sgd1(i) = toc;

    
    if (mod(i,100) == 0)
        fprintf('Cost: %.5f\n', loss_now); 
        fprintf('Grad: %.5f\n', norm(gradient));
    end
    
    
    % if gradient too small, stop
    if (norm(gradient) < epsilon)
        break;
    end

    w = w - step_size * gradient;
    i = i + 1;
    
    if (loss_now < min_cost)
        min_cost = loss_now;
        best_w = w;
    end
    
    
end

w = best_w;

% evaluate accuracy
[pred] = predict(TestX, TrainingX, w, kernel_variance);
accuracy = sum(pred==TestY) / size(TestX, 1);

% plot cost through time
 
p = plot(time_by_iteration_sgd1, cost_by_iteration_sgd1, 'b-', 'LineWidth', 1); 
xlabel('Time (s)'); 
ylabel(sprintf('Cost'));
title ( 'Stochastic Gradient Descent p = 1 ') ;
saveas(p, 'sgd1.png', 'png');

clear TrainingX TraniningY TestX TestY;
save sgd1.mat;

