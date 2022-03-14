N = 100;
iter = 100;

A = randn(N);
A(rand(N)>0.01) = 0;
A(eye(N) == 1) = 0;
sr = max(abs(eig(A)));
A = 0.95 * A/sr;
A = sparse(A);

b = rand(N, 1);

[rowI, colI, v] = find(A);
rowII = [rowI;[1:N]'];
colII = [colI;[1:N]'];
Ival = -ones(N,1);
JA = A; 

%%
tic
hold all
y = zeros(N, iter);

for i = 2:iter
   h = A * y(:, i-1) + b;
   y(:, i) = activation(h);  
end

toc

%%
tic
newtonsIter = sqrt(iter);
netwtonsY = zeros(N, newtonsIter);
for i = 2:newtonsIter
    x = netwtonsY(:, i-1);
    xRaw = A*x + b;
    g = activation(xRaw) - x;
    deltaValues = deltaActivation(xRaw);
    J = sparse(rowII, colII, [v .* deltaValues(rowI); Ival], N, N);
    netwtonsY(:, i) = x - J\g; 
end
toc

%%
figure()
scatter(netwtonsY(:,end), y(:, end), 'filled')
