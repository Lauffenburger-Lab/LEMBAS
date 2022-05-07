N = 300:100:1000;
replicates = 10;
iter = 200;
newtonIter = 100;
sparsity = 0.01;
SRtarget = 0.95;
residualTresh = 1e-6;

timeIter = zeros(length(N), replicates);
timeNewton = zeros(length(N), replicates);
stepConvergedIter = zeros(length(N), replicates);
stepConvergedNewton = zeros(length(N), replicates);

breakOnConvergence = 5;

for i = 1:length(N)
    disp(i)
    for j = 1:replicates
        A = generateSparseMatrix(N(i), sparsity, SRtarget);
        b = rand(N(i), 1);
        x0 = zeros(N(i), 1);
        
        %calculate with iterations
        tic
        y = solveWithIter(A, b, iter, breakOnConvergence, x0);
        timeIter(i, j) = toc;
        stepConvergedIter(i,j) = stepConverged(y, residualTresh);
        
        %calculate with newtons method
        
        %This can be computed once so we will not include it in the timeing
        [rowI, colI, rowII, colII, v, Ival] = getSparsityInfo(A);
        
        tic
        netwtonsY = solveWithNewton(A, b, newtonIter, rowI, rowII, colII, v, Ival, breakOnConvergence, x0);
        %netwtonsY = solveWithLeastSquare(A, b, x0);
        timeNewton(i,j) = toc;
        stepConvergedNewton(i,j) = stepConverged(netwtonsY, residualTresh);
       
        
        same  = sum((y(:, end)-netwtonsY(:, end)).^2)<residualTresh;
        if same == false
           disp('different solutions')
           %figure()
           %plot(y(:, end), netwtonsY(:,end), 'o')
        end
    end
end

%%
hold all
errorbar(N, mean(timeIter'), std(timeIter'))
errorbar(N, mean(timeNewton'), std(timeNewton'))
legend({'Iteration', 'Newton'}, 'location', 'nw')
set(gca, 'YScale', 'log')
xlabel('matrix size [N]')
ylabel('time [s]')

sampleNames = arrayfun(@num2str, 1:size(timeNewton,2), 'UniformOutput', false);        
sampleNames = strcat('Rep_', sampleNames);

conditionNames = arrayfun(@num2str, N, 'UniformOutput', false); 
conditionNames = strcat('N_', conditionNames);

T = array2table(timeIter, 'VariableNames', sampleNames, 'RowNames', conditionNames);
writetable(T, '../Model/figures/SI Figure 3/A_iter.tsv', 'FileType', 'text', 'delim', '\t', 'WriteRowNames', true)

T = array2table(timeNewton, 'VariableNames', sampleNames, 'RowNames', conditionNames);
writetable(T, '../Model/figures/SI Figure 3/A_newton.tsv', 'FileType', 'text', 'delim', '\t', 'WriteRowNames', true)

% figure()
% hold all
% errorbar(N, mean(stepConvergedIter'), std(stepConvergedIter'))
% errorbar(N, mean(stepConvergedNewton'), std(stepConvergedNewton'))
% legend({'Iteration', 'Newton'}, 'location', 'nw')
% xlabel('matrix size [N]')
% ylabel('#Steps untill converged')
% ylim([0, max(stepConvergedIter(:))])


