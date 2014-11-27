function [error_train, error_cv, steps] = learningCurve(X, y, Xcv, ycv, lambda, step)
	% Returns the accuracy error vectors for the training set and the CV set (to be plotted
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
	preds = predict(theta, X(1:steps(i),:));
	error_train(i) = mean(double(preds ~= y(1:steps(i),:)));
	preds_cv = predict(theta, Xcv);
	error_cv(i) = mean(double(preds_cv ~= ycv));
	end
end