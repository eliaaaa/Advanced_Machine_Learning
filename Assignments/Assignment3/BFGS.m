clear;
load bfgs_prep.mat;

epsilon = 0.005;
step = 0.01;
w = zeros(N, 1);
cost_by_iteration_bfgs = [];
time_by_iteration_bfgs = [];

lambda = 1e-3;
H = eye(N, N);

gamma = 0;
delta = 0;
tic;

loss_now = loss_function(TrainingY, gram, w, lambda);
gradient = loss_gradient(TrainingY, gram, w, lambda);
cost_by_iteration_bfgs(1) = loss_now;
time_by_iteration_bfgs(1) = toc;
i = 2;

while(i < 20000)
    p = H * gradient;
    w = w - step * p;
    
    grad_last = gradient;
    
    loss_now = loss_function(TrainingY, gram, w, lambda);
    gradient = loss_gradient(TrainingY, gram, w, lambda);
    cost_by_iteration_bfgs(i) = loss_now;
    time_by_iteration_bfgs(i) = toc;
    
    gamma = gradient - grad_last;
    delta = - step * p;
    
    delta_H = (H*gamma*delta' + delta*gamma'*H) / dot(H*gamma, gamma) ...
        - (1 + dot(gamma, delta)/dot(H*gamma, gamma)) * ...
        ((H*gamma*gamma'*H)/dot(H*gamma, gamma));
    
    if (mod(i,5) == 0)
        fprintf('Cost: %.5f\n', loss_now); 
        fprintf('Grad: %.5f\n', norm(gradient));
    end
    
    if (norm(gradient) < epsilon)
        break;
    end
    
    H = H + delta_H;
    i = i + 1;

end

[pred] = predict(TestX, TrainingX, w, delta2);
accuracy = sum(pred==TestY) / size(TestX, 1);

pl = plot(time_by_iteration_bfgs, cost_by_iteration_bfgs, 'b-', 'LineWidth', 1); 
xlabel('Time (s)'); 
ylabel(sprintf('Cost'));
title ( 'BFGS ') ;
saveas(pl, 'bfgs.png', 'png');

clear TrainingX TraniningY TestX TestY;
save BFGS.mat;