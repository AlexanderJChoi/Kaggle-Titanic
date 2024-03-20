# Titanic Machine Learning Report

## Introduction

This project is a submission for the Kaggle Competition titled '[Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/overview)'.
The Competition provides some data about Titanic passengers. Competition Participants train a Machine Learning Model to predict passengers' status as Survived or Not Survived. 

## Data 

#### The Data contains 11 parameters: 

 - Survived: Whether or not the passenger survived the sinking of the Titanic
 - Pclass: Which class of ticket the passenger purchased (i.e. 1st, 2nd, 3rd class)
 - Name: The name of the passenger
 - Sex: The sex of the passenger. Simplified to male / female
 - Age: The age in years of the passenger
 - SibSp: The number of siblings or spouses the passenger has that are also aboard the Titanic
 - Parch: The number of parents or children the passender has that are also aboard the Titanic
 - Ticket: The passenger's ticket code and number
 - Fare: The amount of money the passenger paid for their ticket
 - Cabin: The cabin codes and numbers for the passenger
 - Embarked: The port where the passenger embarked from
 
In our analysis, we omitted the Name, Ticket, and Cabin parameters. 
Many of the entries in the training data do not contain any Cabin information, and 
we do not believe that Name and Ticket have any relationship with the passenger's rate of survival.

We converted many of the categorical parameters using a one-hot-encoding scheme. 

As part of the Competition, the data is partitioned into two sets: a training set, and a testing set.
The training set contains 891 entries. The testing set contains 418 entries, and does not contain any Survived information for the passengers.

## Machine Learning Methods

We processed the data in [Python 3.10.12](https://www.python.org/) using [NumPy](https://numpy.org/) and [Pandas](https://pandas.pydata.org/), and trained the Machine Learning Models using [SciKitLearn](https://scikit-learn.org/stable/index.html). 

We used two preprocessing steps: Scaling, and Polynomial Featuring. The Scaling step converts each value for a parameter into its corresponding z-score. The Polynomial Featuring step generates higher degree parameters that represent products of the input parameters. 
We included this step because some of the real-valued parameters appeared to have a non-linear relationship with survival rate in the training data.

We utilized two models for prediction: a Logistic Regression Model, and a K-Nearest-Neighbor Model. We performed a hyperparameter "grid" search to determine the optimal hyperparameter setting for each model and the degree of the Polynomial Features step. 
Each setting of the hyperparameters was tested using cross-validation on the training data. 

In the end, the model and hyperparameters that scored the best on the training data was used to predict the survival on the passengers in the testing data.

## Results

Our best result came from a Logistic Regression Model, with degree 2 Polynomial Features. On the training data, this model correctly predicted the Survival of passengers 83% of the time. On the testing data, this model correctly predicted the Survival of passengers 77% of the time.
