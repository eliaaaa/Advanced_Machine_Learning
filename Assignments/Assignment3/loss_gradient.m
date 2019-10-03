function [grad] = loss_gradient(label, gram, w, lambda)
    grad = 2*lambda*w - (gram') *(label.*(1-sigmoid((gram*w).*label)))/size(label, 1);
end

function [sig] = sigmoid(input)
    sig = 1 ./ (1+exp(-input));
end
