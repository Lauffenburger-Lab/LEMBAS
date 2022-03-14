function grad = solveWithLinSolveBackProp(xRaw, gradIn, rowI, colI, rowII, colII, v, Ival)
    N = size(xRaw,1);
    deltaX = deltaActivation(xRaw);
    DL = deltaX .* gradIn;
    Amod = sparse(rowII, colII, [-v .* deltaX(rowI); Ival], N, N);
    %grad = linsolve(full(Amod), DL);
    grad = Amod\DL;
end
