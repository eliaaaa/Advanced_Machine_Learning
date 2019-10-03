clear;
load gram.mat;

% Gradient Descent

step_size = 0.002; 
epsilon = 0.05; 
lambda = 1e-3;
w = zeros(N,1); 
cost_by_iteration_gd = [];
time_by_iteration_gd = [];
i = 1;
tic ;

while(i<20000)
    % compute the cost and gradient of the current w
    
    loss_now = loss_function(TrainingY, Gram_matrix, w, lambda);
    gradient = loss_gradient(TrainingY, Gram_matrix, w, lambda);
    
    cost_by_iteration_gd(i) = loss_now;
    time_by_iteration_gd(i) = toc;
    
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
    
end

[pred] = predict(TestX, TrainingX, w, kernel_variance);
accuracy = sum(pred==TestY) / size(TestX, 1);

% plot cost through time
 
p = plot(time_by_iteration_gd, cost_by_iteration_gd, 'b-', 'LineWidth', 1); 
xlabel('Time (s)'); 
ylabel(sprintf('Cost'));
title ( 'Gradient Descent ') ;
saveas(p, 'gd.png', 'png');

clear TrainingX TraniningY TestX TestY;
save gd.mat;



