function y = solveWithIter(A, b, iter, breakOnConverge, x0)
    tresh = 1e-6;
    N = size(A, 1);
    y = zeros(N, iter);
    y(:,1) = x0;
    for i = 2:iter
       h = A * y(:, i-1) + b;
       y(:, i) = activation(h);  

       if i>breakOnConverge
            converged = sum(abs(y(:, i-1)-y(:,i)))<tresh; %expensive, but probably worth it
            if converged
                for j = (i+1):iter
                   y(:,j) = y(:, i); %fill out matrix
                end
                break   
            end
       end
    end
end

