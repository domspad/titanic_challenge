%% Initialization
clear ; close all; clc


%% Load Data
%  The first two columns contains the exam scores and the third column
%  contains the label
data = load('train1.txt');
Xtrain = data(1:691, 1:9); ytrain = data(1:691, 10);
Xcv = data(692:end,1:9); ycv = data(692:end,10);

%% ============ Part 2: Compute Cost and Gradient ============

%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(Xtrain);

% Add intercept term to x and X_test
Xtrain = [ones(m, 1) Xtrain];
Xcv = [ones(size(Xcv,1),1) Xcv];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, Xtrain, ytrain);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);


%% ============= Part 3: Optimizing using fminunc  =============


options = optimset('GradObj', 'on', 'MaxIter', 400);

[theta, cost] = ...
	fminunc(@(t)(costFunction(t, Xtrain, ytrain)), initial_theta, options);

fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

%% ============== Part 4: Predict and Accuracies ==============

% Compute accuracy on our training set
ptrain = predict(theta, Xtrain);
fprintf('Train Accuracy: %f\n', mean(double(ptrain == ytrain)) * 100);
fprintf('Train cost %f\n', costFunction(theta, Xtrain, ytrain));;


pcv = predict(theta, Xcv);
fprintf('CV Accuracy: %f\n', mean(double(pcv == ycv)) * 100);
fprintf('CV cost %f\n', costFunction(theta, Xcv, ycv));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============= Part 5: plotting Learning Curve ==============

[error_train, error_cv] = learningCurve(Xtrain, ytrain, Xcv, ycv);

plot(1:30:m, error_train, 1:30:m, error_cv);
title('Learning curve for logistic regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
% axis([0 13 0 150])

fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:30:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_cv(i));
end



