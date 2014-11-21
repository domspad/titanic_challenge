function [error_train, error_cv] = learningCurve(X, y, Xcv, ycv)
	% Returns the error vectors for the training set and the CV set (to be plotted
	% as the y-vals of a learning curve) 

m = size(X, 1); % number of train examples

error_train = zeros(m, 1);
error_cv = zeros(m, 1);

for i=1:m 
	%train the model
	theta = trainLogReg(X(1:i,:), y(1:i,:), 0);

	%calculate and save the errors
	error_train(i) = costFunction(theta, X(1:i,:), y(1:i,:));
	error_cv(i) = costFunction(theta, Xcv, ycv);

	%FOR RETURNING ACCURACIES...
%	pred = predict(theta, X(1:i,:));
%	error_train(i) = mean(double(pred == y(1:i,:))) * 100
%	predcv = predict(theta, Xcv);
%	error_cv = predict(theta, X)
	end

end