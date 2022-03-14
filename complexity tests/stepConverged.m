function passTresh = stepConverged(y, tresh)
    residual = sum((y-y(:,end)).^2);
    passTresh = min(find(residual<tresh));
end

