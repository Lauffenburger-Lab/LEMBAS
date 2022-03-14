function [rowI, colI, rowII, colII, v, Ival] = getSparsityInfo(A)
    N = size(A, 1);
    [rowI, colI, v] = find(A);
    rowII = [rowI;[1:N]'];
    colII = [colI;[1:N]'];
    Ival = -ones(N,1);
end

