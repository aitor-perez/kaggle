# Execute after the preprocessing script

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

pd.options.display.max_columns = 20

#%% Read preprocessed data
train_data = pd.read_csv('preprocessed_train.csv')
test_data = pd.read_csv('preprocessed_test.csv')

X_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived']
X_test = test_data.drop('PassengerId', axis=1)
print(X_train.shape, y_train.shape, X_test.shape)

#%% Prepare classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(),
    'Support Vector Machines': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3),
    'Gaussian Naive Bayes': GaussianNB(),
    'Perceptron': Perceptron(),
    'Linear SVC': LinearSVC(max_iter=3000),
    'Stochastic Gradient Descent': SGDClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100)
}

#%% Prepare structure to store the scores for each classifier
scores = pd.DataFrame(classifiers.keys(), columns=['Model'])
scores['Score'] = pd.Series(np.zeros(len(classifiers)))

print(scores)

#%% Training of classifiers
for model, clf in classifiers.items():
    clf.fit(X_train, y_train)
    score = clf.score(X_train, y_train)
    scores.loc[scores['Model'] == model, 'Score'] = score

scores = scores.sort_values(by='Score', ascending=False)
print(scores)

best_model = scores.loc[scores['Score'].idxmax(), 'Model']
best_classifier = classifiers[best_model]

#%% Prediction
y_pred = best_classifier.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': y_pred})
output.to_csv('predictions.csv', index=False)
print("Submission successfully saved!")

