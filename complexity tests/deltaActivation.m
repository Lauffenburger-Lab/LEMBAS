function y = deltaActivation(x)
    y = ones(size(x));
    y(x<0) = 0.01;
    filter = x>0.5;
    y(filter) = 0.25./(x(filter).^2);
end