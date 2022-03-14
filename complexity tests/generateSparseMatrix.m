function A = generateSparseMatrix(N, sparsity, SRtarget)
    A = randn(N);
    A(rand(N)>sparsity) = 0;
    A(eye(N) == 1) = 0;
    sr = max(abs(eig(A)));
    A = SRtarget * A/sr;
    A = sparse(A);
end

