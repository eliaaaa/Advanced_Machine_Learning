clear;
load bfgs_prep.mat;

epsilon = 1e-5;
step = 0.001;
w = zeros(N, 1);
lambda = 1e-6;
H = eye(N, N);

ep = 0;
cost_by_iteration_lbfgs = [];
time_by_iteration_lbfgs = [];

m = 10;
gamma = zeros(N, 1);
delta = zeros(N, 1);
rho = zeros(1, m);
gradients = zeros(N, 1);
alpha = zeros(1, m);
beta = zeros(1, m);
tic;

loss_now = loss_function(TrainingY, gram, w, lambda);
gradient = loss_gradient(TrainingY, gram, w, lambda);

cost_by_iteration_lbfgs(end+1) = loss_now;
time_by_iteration_lbfgs(end+1) = toc;
p = H * gradient;

while(toc < 1000)
    w = w - step * p;
    
    grad_last = gradient;
    
    loss_now = loss_function(TrainingY, gram, w, lambda);
    gradient = loss_gradient(TrainingY, gram, w, lambda);
    cost_by_iteration_lbfgs(end+1) = loss_now;
    time_by_iteration_lbfgs(end+1) = toc;
    
    if ep > m-1
        if ep > m
            gradients = gradients(:, 2:size(gradients,2));
        end
        gamma = gamma(:, 2:size(gamma,2));
        delta = delta(:, 2:size(delta,2));
        rho = rho(2:size(rho));
        L = m;
    else
        L = ep;
    end
     
    if ep == 0
        gamma = gradient - grad_last; 
        delta = - step * p; 
        rho = 1 ./ (gamma'*delta);
        gradients = gradient;
    else
        gamma = cat(2, gamma, gradient - grad_last);
        delta = cat(2, delta, - step * p);
        rho = cat(1, rho, 1 ./ (gamma(:, end)'*delta(:, end)));
        gradients = cat(2, gradients, gradient);
    end
        
    for i = L:-1:1
        alpha(i) = rho(i)*delta(:, i)'*gradients(:, i+1);
        gradients(:, i) = gradients(:, i+1) - alpha(i)*gamma(:, i);
    end
    for i = 1:L
        beta(i) = rho(i)*gamma(:, i)'*p;
        p = p + (alpha(i)-beta(i))*delta(:, i);
    end
    
    ep = ep + 1;
    
    if (mod(ep,1000) == 0)
        fprintf('Cost: %.5f\n', loss_now); 
        fprintf('Grad: %.5f\n', norm(gradient));
    end
    
    if (norm(gradient) < epsilon)
        break;
    end
    
end

[pred] = predict(TestX, TrainingX, w, delta2);
accuracy = sum(pred==TestY) / size(TestX, 1);

pl = plot(time_by_iteration_lbfgs, cost_by_iteration_lbfgs, 'b-', 'LineWidth', 1); 
xlabel('Time (s)'); 
ylabel(sprintf('Cost'));
title ( 'LBFGS ') ;
saveas(pl, 'lbfgs.png', 'png');


clear TrainingX TraniningY TestX TestY;
save LBFGS.mat;