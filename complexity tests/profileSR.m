N = 500;
replicates = 10;
iter = 200;
newtonIter = 200;
sparsity = 0.01;
SRtarget = linspace(1, 5, 10);
residualTresh = 1e-6;

stepConvergedIter = zeros(length(N), replicates);
stepConvergedNewton = zeros(length(N), replicates);
wasSolutionSame = zeros(length(N), replicates);
defactoSR = zeros(length(N), replicates);
%defactoSRJacobian = zeros(length(N), replicates);

for i = 1:length(SRtarget)
    disp(i)
    for j = 1:replicates
        x0 = rand(N, 1);
        A = generateSparseMatrix(N, sparsity, SRtarget(i));
        b = rand(N, 1);
        
        y = solveWithIter(A, b, iter, 5, x0);
        stepConvergedIter(i,j) = stepConverged(y, residualTresh);
        xRaw = A*y(:,end) + b;       
        [rowI, colI, rowII, colII, v, Ival] = getSparsityInfo(A);
        
        netwtonsY = solveWithNewton(A, b, newtonIter, rowI, rowII, colII, v, Ival, 5, x0);
        stepConvergedNewton(i,j) = stepConverged(netwtonsY, residualTresh);
        
        wasSolutionSame(i, j) = sum((y(:, end)-netwtonsY(:,end)).^2)<residualTresh;
        
        deltaValues = deltaActivation(xRaw);
        J = sparse(rowI, colI, v .* deltaValues(rowI), N, N);
        defactoSR(i,j) = max(abs(eig(full(J))));        
        %J = sparse(rowII, colII, [v .* deltaValues(rowI); -Ival], N, N);
        %defactoSRJacobian(i,j) = max(abs(eig(full(J))));     
    end
end

%%
hold all
%errorbar(SRtarget, mean(stepConvergedIter'), std(stepConvergedIter'))
%errorbar(SRtarget, mean(stepConvergedNewton'), std(stepConvergedNewton'))
scatter(defactoSR(:), stepConvergedIter(:), 'filled');
scatter(defactoSR(:), stepConvergedNewton(:), 'filled');

sr = linspace(min(defactoSR(:)), 0.98);
estimatedN = log(1e-4)./log(sr);
plot(sr, estimatedN, 'k-')

legend({'Iteration', 'Newton', 'theory'}, 'location', 'nw')
xlabel('Estimated spectral radius at steady state')
ylabel('#Steps untill converged')
ylim([0, newtonIter])


figure()
hold all
plot(SRtarget, sum(stepConvergedIter==iter, 2)/replicates, 'o-')
plot(SRtarget, sum(stepConvergedNewton==newtonIter, 2)/replicates, 'o-')
xlabel('Spectral radius of weight matrix')
ylabel('frequency divergence')
legend({'Iteration', 'Newton'}, 'location', 'nw')

