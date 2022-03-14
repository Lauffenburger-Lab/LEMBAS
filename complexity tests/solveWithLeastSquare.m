function y = solveWithLeastSquare(A, b, x0)
    y = lsqnonlin(@(x) step(x, A, b), x0);
end

function delta = step(x, A, b)
    xRaw = A*x + b;
    y = activation(xRaw);    
    delta = y-x;
end