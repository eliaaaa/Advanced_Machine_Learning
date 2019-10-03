function [loss] = loss_function(label, gram, w, lambda)
    loss = lambda*(w'*w) - sum(log(sigmoid((gram*w).*label)))/size(label, 1);
end

function [sig] = sigmoid(input)
    sig = 1 ./ (1+exp(-input));
end
