function [error_train, error_cv, steps] = learningCurve(X, y, Xcv, ycv, lambda, step)
	% Returns the error vectors for the training set and the CV set (to be plotted
	% as the y-vals of a learning curve) 

m = size(X, 1); % number of train examples

%calculate and return the steps at which errors are calculated
steps = step:step:m;
if steps(end) ~= m 
	steps = [steps m];
end

error_train = zeros(length(steps), 1);
error_cv = zeros(length(steps), 1);

for i=1:length(steps)
	%train the model
	theta = trainLogReg(X(1:steps(i),:), y(1:steps(i),:), lambda);

	%calculate and save the errors
	error_train(i) = costFunction(theta, X(1:steps(i),:), y(1:steps(i),:), 0);
	error_cv(i) = costFunction(theta, Xcv, ycv, 0);
	end
end