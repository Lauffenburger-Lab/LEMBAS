function netwtonsY = solveWithNewton(A, b, newtonsIter, rowI, rowII, colII, v, Ival, breakOnConverge, x0)
    tresh = 1e-6;
    N = size(A, 1);
    netwtonsY = zeros(N, newtonsIter);
    netwtonsY(:, 1) = x0;
    
    for i = 2:newtonsIter
        x = netwtonsY(:, i-1);
        xRaw = A*x + b;
        g = activation(xRaw) - x;
        deltaValues = deltaActivation(xRaw);
        J = sparse(rowII, colII, [v .* deltaValues(rowI); Ival], N, N);
        netwtonsY(:, i) = x - J\g;
        

        if i>breakOnConverge    
            converged = sum(abs(netwtonsY(:, i-1)-netwtonsY(:,i)))<tresh;
            if converged
                for j = (i+1):newtonsIter
                   netwtonsY(:,j) = netwtonsY(:, i); %fill out matrix
                end
                break   
            end
        end
    end
end

