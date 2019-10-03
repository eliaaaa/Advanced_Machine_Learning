function [predict] = predict(testX, trainX, w, kernel_variance)

    testX_kernel = zeros(size(testX, 1), size(trainX, 1));
    for i = 1:size(testX, 1)
        for j = 1:size(trainX, 1)
            testX_kernel(i, j) = exp( - (norm(testX(i, :) - trainX(j, :))^2) ...
            / (2 * kernel_variance));
        end
    end
    
    predict = 2 * (sigmoid(testX_kernel*w)>0.5) - 1;
end
    
function [sig] = sigmoid(input)
    sig = 1 ./ (1+exp(-input));
end
