clear;

load gd.mat;
load sgd1.mat;
load sgd100.mat;
load BFGS.mat;
load LBFGS.mat;

plot(time_by_iteration_gd, cost_by_iteration_gd, 'b-', 'LineWidth', 1); 
hold on;
plot(time_by_iteration_sgd1, cost_by_iteration_sgd1, 'r-', 'LineWidth', 1);
hold on;
plot(time_by_iteration_sgd100, cost_by_iteration_sgd100, 'g-', 'LineWidth', 1); 
hold on;
plot(time_by_iteration_bfgs, cost_by_iteration_bfgs, 'y-', 'LineWidth', 1); 
hold on;
plot(time_by_iteration_lbfgs, cost_by_iteration_lbfgs, 'k-', 'LineWidth', 1); 
hold on;

xlabel('Time (s)'); 
title ( 'Cost/Time Comparison ') ;
legend('GD','SGD1', 'SGD100','BFGS','LBFGS');
saveas(gcf,'comparison.png'); 
