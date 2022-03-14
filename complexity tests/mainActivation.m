N = 5;
iter = 20;

A = rand(N);
A(eye(N) == 1) = 0;
sr = max(abs(eig(A)));
A = 0.95 * A/sr;

b = rand(N, 1);


%%
tic
hold all
y = zeros(N, iter);

for i = 2:iter
   h = A * y(:, i-1) + b;
   y(:, i) = activation(h);  
end

plot(y', 'k')
toc

%%
tic
newtonsIter = 10;
netwtonsY = zeros(N, newtonsIter);
for i = 2:newtonsIter
    x = netwtonsY(:, i-1);
    xRaw = A*x + b;
    g = activation(xRaw) - x;
    J = diag(deltaActivation(xRaw)) * A - eye(5);
    netwtonsY(:, i) = x - J\g; 
end
solution = netwtonsY(:,end);
plot(netwtonsY', 'r')
%plot([0 iter], [solution solution])
toc

%%
figure()
scatter(netwtonsY(:,end), y(:, end), 'filled')
