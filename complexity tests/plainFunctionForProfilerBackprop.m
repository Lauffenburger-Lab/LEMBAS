iter = 100;

sr = 0.95;
N = 500;

A = generateSparseMatrix(N, 0.01, sr);
b = rand(N, 1);
targetY = randn(N, 1);

%calculate with iterations
y = solveWithIter(A, b, iter, inf);
Xraw = A*y(:,end) + b;
gradIn = y(:,end)-targetY;
gradIter = solveWithIterBackProp(A, gradIn, Xraw, iter);

[rowI, colI, rowII, colII, v, Ival] = getSparsityInfo(A');
Ival=-Ival;

gradSolve = solveWithLinSolveBackProp(Xraw, gradIn, rowI, colI, rowII, colII, v, Ival);

scatter(gradIter(:, end), gradSolve)