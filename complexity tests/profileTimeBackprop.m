N = 300:100:1000;
%N = [200];
replicates = 10;
iter = 100;
newtonIter = 10;
sparsity = 0.01;
SRtarget = 0.95;
residualTresh = 1e-6;
solutionMatchTresh = 1e-3;

timeIter = zeros(length(N), replicates);
timeLinsolve = zeros(length(N), replicates);
stepConvergedIter = zeros(length(N), replicates);


for i = 1:length(N)
    disp(i)
    for j = 1:replicates
        A = generateSparseMatrix(N(i), sparsity, SRtarget);
        b = rand(N(i), 1);
        targetY = randn(N(i), 1);
        x0 = zeros(N(i), 1);
        y = solveWithIter(A, b, iter, inf, x0);
        Xraw = A*y(:,end) + b;
        
        gradIn = y(:,end)-targetY;
        
        %calculate with iterations
        tic        
        gradIter = solveWithIterBackProp(A, gradIn, Xraw, iter);
        timeIter(i, j) = toc;
        stepConvergedIter(i,j) = stepConverged(gradIter, residualTresh);
        
        %calculate with Linsolve
        
        %This can be computed just once so we will not include it in the timeing
        AT = A';
        [rowI, colI, rowII, colII, v, Ival] = getSparsityInfo(A');
        Ival=-Ival;
        
        tic
        gradSolve = solveWithLinSolveBackProp(Xraw, gradIn, rowI, colI, rowII, colII, v, Ival);
        timeLinsolve(i,j) = toc;
                 
         same  = sum((gradIter(:, end)-gradSolve).^2)<solutionMatchTresh;
         if same == false
            disp('different solutions') 
            %figure()
            %scatter(gradIter(:, end), gradSolve)
         end
    end
end


%%
figure()
hold all
errorbar(N, mean(timeIter'), std(timeIter'))
errorbar(N, mean(timeLinsolve'), std(timeLinsolve'))
legend({'Iteration', 'LinSolve'}, 'location', 'nw')
set(gca, 'YScale', 'log')
xlabel('matrix size [N]')
ylabel('time [s]')

figure()
hold all
errorbar(N, mean(stepConvergedIter'), std(stepConvergedIter'))
plot(N, ones(size(N)));
legend({'Iteration', 'LinSolve'}, 'location', 'nw')
xlabel('matrix size [N]')
ylabel('#Steps untill converged')
ylim([0, max(stepConvergedIter(:))])


