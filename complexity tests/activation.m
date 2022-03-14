function y = activation(x)
    y = x;
    y(x<0) = 0.01 * x(x<0);
    y(x>0.5) = 1 - 0.25./x(x>0.5);
end