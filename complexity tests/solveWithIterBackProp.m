function grad = solveWithIterBackProp(A, gradIn, xRaw, iter)
    AT = A';
    grad = zeros(size(gradIn));
    deltaX = deltaActivation(xRaw);
    
    grad = zeros(length(gradIn), iter);
    for i = 2:iter
        grad(:, i) = deltaX .* (AT * grad(:, i-1) + gradIn);
        grad(:, i) = gradCliping(grad(:, i));
    end
end

function grad = gradCliping(grad)
    n = 5;
    clipingFilter = grad<-n;
    grad(clipingFilter) = tanh(grad(clipingFilter) + n) - n;
    clipingFilter = grad>n;
    grad(clipingFilter) = tanh(grad(clipingFilter) - n) + n;
end