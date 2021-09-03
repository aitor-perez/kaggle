# Execute after the preprocessing script

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

pd.options.display.max_columns = 20

import time

#%% Read preprocessed data
train_data = pd.read_csv('preprocessed_train.csv')
test_data = pd.read_csv('preprocessed_test.csv')

X_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived']
X_test = test_data.drop('PassengerId', axis=1)
print(X_train.shape, y_train.shape, X_test.shape)

#%% Prepare classifiers
classifiers = {
    'Logistic Regression': {
        'params': {'C': np.logspace(-4, 4, 9)},
        'clf': LogisticRegression()
    },
    # 'Support Vector Machines': {
    #     'params': {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'C': np.logspace(-4, 4, 9)},
    #     'clf': SVC()
    # },
    'Support Vector Machines': {
        'params': {'kernel': ['poly', 'rbf'], 'C': np.logspace(-1, 3, 5)},
        'clf': SVC()
    },
    'K-Nearest Neighbors': {
        'params': {'n_neighbors': list(range(3, 20, 2))},
        'clf': KNeighborsClassifier()
    },
    'Gaussian Naive Bayes': {
        'params': {'var_smoothing': np.logspace(-9, 0, 10)},
        'clf': GaussianNB()
    },
    'Perceptron': {
        'params': {'alpha': np.logspace(-4, -0.5, 8)},
        'clf': Perceptron()
    },
    'Linear SVC': {
        'params': {'C': np.logspace(-4, 4, 9)},
        'clf': LinearSVC()
    },
    'Stochastic Gradient Descent': {
        'params': {'penalty': ['l1', 'l2'], 'alpha': np.logspace(-4, -0.5, 8)},
        'clf': SGDClassifier()
    },
    'Decision Tree': {
        'params': {'criterion': ['gini', 'entropy'], 'max_depth': list(range(5, 35, 5))},
        'clf': DecisionTreeClassifier()
    },
    'Random Forest': {
        'params': {'n_estimators': list(range(11, 32, 4)), 'criterion': ['gini', 'entropy'], 'max_depth': list(range(5, 25, 5))},
        'clf': RandomForestClassifier()
    }
}


#%% Prepare structure to store the scores for each classifier
scores = pd.DataFrame(classifiers.keys(), columns=['Model'])
scores['Score'] = pd.Series(np.zeros(len(classifiers)))

print(scores)

#%% Training of classifiers with cross validation
cv = KFold(n_splits=10, shuffle=True, random_state=0)

for model, classifier in classifiers.items():
    print("Tuning", model)
    start = time.time()

    classifier['clf'] = GridSearchCV(classifier['clf'], classifier['params'], cv=cv, scoring='accuracy')
    classifier['clf'].fit(X_train, y_train)
    score = classifier['clf'].best_score_
    scores.loc[scores['Model'] == model, 'Score'] = score

    stop = time.time()
    duration = stop - start
    print("Best params:", classifier['clf'].best_params_)
    print("Score:", score)
    print("Finished tuning", model, "Total time:", round(duration, 2), "s")


scores = scores.sort_values(by='Score', ascending=False)
print(scores)

#%% Prediction of best model
best_model = scores.loc[scores['Score'].idxmax(), 'Model']
best_clf = classifiers[best_model]['clf']

y_pred = best_clf.predict(X_test)

#%% Prediction with majority vote

y_preds = pd.DataFrame()
for model, classifier in classifiers.items():
    y_preds[model] = classifier['clf'].predict(X_test)

y_preds['Sum'] = y_preds.sum(axis=1)
y_preds['Majority'] = y_preds.mode(axis=1)

y_pred = y_preds['Majority']

#%% Save predictions
output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': y_pred})
output.to_csv('predictions.csv', index=False)
print("Submission successfully saved!")

