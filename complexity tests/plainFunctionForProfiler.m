newtonIter = 500;
iter = 500;

sr = 5;
N = 500;

A = generateSparseMatrix(N, 0.01, sr);
b = rand(N, 1);

%calculate with iterations
y = solveWithIter(A, b, iter, inf);


%calculate with newtons method

[rowI, colI, rowII, colII, v, Ival] = getSparsityInfo(A);
netwtonsY = solveWithNewton(A, b, newtonIter, rowI, rowII, colII, v, Ival, inf);


scatter(y(:, end), netwtonsY(:,end))

residualTresh = 1e-5;
stepConverged(y, residualTresh)
stepConverged(netwtonsY, residualTresh)