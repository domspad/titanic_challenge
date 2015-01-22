#Titanic Challange
##The data analysis cycle
=================

This repository is a practice of the steps one should take when applying any machine learning model to a prediction problem. They are taken from Andrew Ng's Coursera Machine Learning course. Also, it is a practice in implementing logistic regularized logistic regression, rather than using one say from sci-kit learn.

###The Problem

Predicting whether a passenger of the Titanic survived based on basic demographic data about them

[The challenge on Kaggle](https://www.kaggle.com/c/titanic-gettingStarted)

###Data analysis cycle


1.  Train a simple model on the training data (e.g. logistic regression) and look at score achieved on the CV set. 

2.  Plot learning curves to determine whether model has high bias or high variance. 

3.  If high bias, then add features to the data by looking at where the most errors were made in the CV set ("error analysis"). Otherwise, the model has high variance and instead look into either regularizing or removing some features.

4.  Repeat 1-3 with resulting model

###Results

In terms of prediction accuracy

|         | train | CV | test |
|---------|-------|----|------|
| alldead |   61.3    |  62.5  |   62.5   |
| model1  |   78.0    |  80.0  |   77.0   |
| model2  |   82.5    |  85.5  |  78.8    |

Both models used logistic regression, model1 has minimal preprocessing done to the data (total of 9 features), while for model2 I analyzed the errors model1 made on the CV dataset and tried to add additional features that would be helpful in identifying those cases. 

As a result, the CV set saw the most gains in prediction accuracy, which is not surprising considering the features were selected based on errors originally made on that set, however, the test set surprisingly didn't see much improvement from the additional 20 features. I think this may be do to now there being too many features, and thus overfitting the training set.
