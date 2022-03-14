N = 5;
iter = 100;

A = rand(N);
A(eye(N) == 1) = 0;
sr = max(abs(eig(A)));
A = 0.95 * A/sr;

b = rand(N, 1);


%%
hold all
y = zeros(N, iter);

for i = 2:iter
   y(:, i) = A * y(:, i-1) + b;  
end

plot(y')

%%
newtonsIter = 10;
netwtonsY = zeros(N, newtonsIter);
for i = 2:10
    x = netwtonsY(:, i-1);
    g = A*x + b - x;
    netwtonsY(:, i) = x - inv(A-eye(5))  * g; 
end
solution = netwtonsY(:,2);
plot([0 iter], [solution solution])