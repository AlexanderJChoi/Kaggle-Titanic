import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, ParameterGrid
from sklearn.neighbors import KNeighborsClassifier

## Read Training Data + Filter
print("Reading Data ..")
df = pd.read_csv('train.csv', index_col=0)

df['Pclass'] = df['Pclass'].astype('object')
Pclass_onehot = pd.get_dummies(df[['Pclass']])
df = pd.concat([df, Pclass_onehot], axis=1)

Sex_onehot = pd.get_dummies(df[['Sex']])
df = pd.concat([df, Sex_onehot], axis=1)

Embarked_onehot = pd.get_dummies(df[['Embarked']])
df = pd.concat([df, Embarked_onehot], axis=1)

df.drop(labels=['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

df['Age'] = df['Age'].replace(np.nan, df['Age'].mean())

train_X = df.drop(labels=['Survived'], axis=1)
train_Y = df[['Survived']]

## Set up Classification Models

print("Setting Up Models")

## Setup Logistic Regression
# Standard Scaler
# Polynomial Features
# Logistic Regression
 
transforms = [ [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(degree = 2, include_bias=False)), ('model', LogisticRegression(max_iter=10000000))] ]
param_grids = [ ParameterGrid({'polynomial__degree': [1, 2, 3, 4], 'polynomial__include_bias': [True, False], 'model__C': [100, 10, 1, 0.1, 0.01, 0.001], 'model__solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']}) ]

## Setup KNN Classification
# standard scaler

transforms += [ [('scale', StandardScaler()), ('polynomial', PolynomialFeatures()), ('model', KNeighborsClassifier())] ]
param_grids += [ ParameterGrid({'polynomial__degree': [1, 2, 3, 4], 'polynomial__include_bias': [True, False], 'model__n_neighbors': [1, 5, 10, 15, 20], 'model__weights': ['uniform', 'distance'], 'model__p': [1, 2, 3, 4]}) ]

best_param = 0
best_score = 0
best_pipe = 0

pipe = Pipeline([])
for t in range(len(transforms)):
	pipe = Pipeline(transforms[t])
	print (f"Testing Model: {str(pipe)}")
	print ("# Selecting Hyperparameters")
	for p in param_grids[t]:
		## Cross Validation
		print (f"## Cross Validation Scores: ({p})")
		cv_results = cross_val_score(pipe.set_params(**p), train_X, train_Y['Survived'], scoring='accuracy') 	# NOTE: ** is "unpacking" the dictionary elements for the parameter list
		score = cv_results.mean()
		print (f"### {cv_results} : mean : {cv_results.mean()}")
		if score > best_score:
			best_score = score
			best_param = p
			best_pipe = pipe

print (f"Best Model: {str(best_pipe)}")
print(f"Best Parameters: \n {best_param}\n Score: {best_score}") 
pipe = best_pipe
pipe.set_params(**best_param)

print("Training Model")

pipe.fit(train_X, train_Y['Survived'])

## Read Testing Data + Filter
print("Reading Test Data")
tf = pd.read_csv('test.csv')

test_labels = tf['PassengerId']

tf['Pclass'] = tf['Pclass'].astype('object')
Pclass_onehot_t = pd.get_dummies(tf[['Pclass']])
tf = pd.concat([tf, Pclass_onehot_t], axis=1)

Sex_onehot_t = pd.get_dummies(tf[['Sex']])
tf = pd.concat([tf, Sex_onehot_t], axis=1)

Embarked_onehot_t = pd.get_dummies(tf[['Embarked']])
tf = pd.concat([tf, Embarked_onehot_t], axis=1)

tf.drop(labels=['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'PassengerId'], axis=1, inplace=True)

tf['Age'] = tf['Age'].replace(np.nan, tf['Age'].mean())
tf['Fare'] = tf['Fare'].replace(np.nan, tf['Fare'].mean())

## Test model
print("Testing Model")
test_results = pd.DataFrame(pipe.predict(tf), columns = ['Survived'])
test_results = pd.concat([test_labels, test_results], axis=1)
print(test_results)
test_results.to_csv("results.csv", index=False)
