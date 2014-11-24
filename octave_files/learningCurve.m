function [error_train, error_cv] = learningCurve(X, y, Xcv, ycv)
	% Returns the error vectors for the training set and the CV set (to be plotted
	% as the y-vals of a learning curve) 

m = size(X, 1); % number of train examples
step = 30;
num = ceil(m/step);

error_train = zeros(num, 1);
error_cv = zeros(num, 1);

for i=1:num
	%train the model
	theta = trainLogReg(X(1:(i-1)*step+1,:), y(1:(i-1)*step+1,:), 0);

	%calculate and save the errors
	error_train(i) = costFunction(theta, X(1:(i-1)*step+1,:), y(1:(i-1)*step+1,:));
	error_cv(i) = costFunction(theta, Xcv, ycv);

	%FOR RETURNING ACCURACIES...
%	pred = predict(theta, X(1:i,:));
%	error_train(i) = mean(double(pred == y(1:i,:))) * 100
%	predcv = predict(theta, Xcv);
%	error_cv = predict(theta, X)
	end

end