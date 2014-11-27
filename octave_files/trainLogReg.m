function [theta] = trainLogReg(X,y,lambda)
	%returns the optimized parameter 'theta' for logistic regression

initial_theta = zeros(size(X,2),1);

options = optimset('GradObj', 'on', 'MaxIter', 200);

theta = fminunc(@(t)(costFunction(t, X, y, lambda)), initial_theta, options);

end