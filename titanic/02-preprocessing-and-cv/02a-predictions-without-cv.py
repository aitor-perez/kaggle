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

#%% Prepare structure to store accuracy scores
models = ['Logistic Regression',
          'Support Vector Machines',
          'K-Nearest Neighbors',
          'Gaussian Naive Bayes',
          'Perceptron',
          'Linear SVC',
          'Stochastic Gradient Descent',
          'Decision Tree',
          'Random Forest']
scores = pd.DataFrame(np.array(models), columns=['Model'])
scores['Score'] = pd.Series(np.zeros(len(models)))

print(scores)

#%% Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
acc_logreg = logreg.score(X_train, y_train)
scores.loc[scores['Model'] == 'Logistic Regression', 'Score'] = acc_logreg

print(scores.sort_values(by='Score', ascending=False))

# Logistic Regression coefficients
logreg_coeffs = pd.DataFrame(train_data.columns.delete(0), columns=['Feature'])
logreg_coeffs['Correlation'] = pd.Series(logreg.coef_[0])
logreg_coeffs = logreg_coeffs.sort_values(by='Correlation', ascending=False)
# print(logreg_coeffs)

#%% Support Vector Machines
svc = SVC()
svc.fit(X_train, y_train)
acc_svc = svc.score(X_train, y_train)
scores.loc[scores['Model'] == 'Support Vector Machines', 'Score'] = acc_svc

print(scores.sort_values(by='Score', ascending=False))

#%% K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
acc_knn = knn.score(X_train, y_train)
scores.loc[scores['Model'] == 'K-Nearest Neighbors', 'Score'] = acc_knn

print(scores.sort_values(by='Score', ascending=False))

#%% Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
acc_gnb = gnb.score(X_train, y_train)
scores.loc[scores['Model'] == 'Gaussian Naive Bayes', 'Score'] = acc_gnb

print(scores.sort_values(by='Score', ascending=False))

#%% Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, y_train)
acc_perceptron = perceptron.score(X_train, y_train)
scores.loc[scores['Model'] == 'Perceptron', 'Score'] = acc_perceptron

print(scores.sort_values(by='Score', ascending=False))

#%% Linear SVC
lin_svc = LinearSVC(max_iter=3000)
lin_svc.fit(X_train, y_train)
acc_lin_svc = lin_svc.score(X_train, y_train)
scores.loc[scores['Model'] == 'Linear SVC', 'Score'] = acc_lin_svc

print(scores.sort_values(by='Score', ascending=False))

#%% Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, y_train)
acc_sgd = sgd.score(X_train, y_train)
scores.loc[scores['Model'] == 'Stochastic Gradient Descent', 'Score'] = acc_sgd

print(scores.sort_values(by='Score', ascending=False))

#%% Decision Tree
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
acc_dtc = dtc.score(X_train, y_train)
scores.loc[scores['Model'] == 'Decision Tree', 'Score'] = acc_dtc

print(scores.sort_values(by='Score', ascending=False))

#%% Random Forest
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
acc_rfc = rfc.score(X_train, y_train)
scores.loc[scores['Model'] == 'Random Forest', 'Score'] = acc_rfc

print(scores.sort_values(by='Score', ascending=False))

#%% Final prediction

# Both Decision Tree and Random Forest have the best Accuracy.
# We choose to use Random Forest as it better avoids overfitting.

y_pred = rfc.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': y_pred})
output.to_csv('predictions.csv', index=False)
print("Submission successfully saved!")

