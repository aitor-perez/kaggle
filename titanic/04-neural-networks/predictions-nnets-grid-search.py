# Execute after the preprocessing script

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPClassifier

pd.options.display.max_columns = 20

import time

#%% Read preprocessed data
train_data = pd.read_csv('preprocessed_train.csv')
test_data = pd.read_csv('preprocessed_test.csv')

X_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived']
X_test = test_data.drop('PassengerId', axis=1)
print(X_train.shape, y_train.shape, X_test.shape)

#%% Training of nnet with cross validation
cv = KFold(n_splits=10, shuffle=True)

nnet = MLPClassifier(solver='lbfgs', max_iter=10000)

params = {
    'hidden_layer_sizes': [(n,) for n in range(3, 17, 2)],
    'activation': ['relu'],
    'alpha': np.logspace(-5, -3, 3)
}

# params = {
#     'hidden_layer_sizes': [(5,)],
#     'activation': ['relu'],
#     'alpha': [0.0001]
# }

print("Tuning Neural Network")
start = time.time()

nnet = GridSearchCV(nnet, params, cv=cv, scoring='accuracy', verbose=3)
nnet.fit(X_train, y_train)
score = nnet.best_score_

stop = time.time()
duration = stop - start

print("Best params:", nnet.best_params_)
print("Score:", score)
print("Finished tuning Neural Network. Total time:", round(duration, 2), "s")

print('*'*40)
print('Weights:', [coef.shape for coef in nnet.best_estimator_.coefs_])
print('*'*40)
print('Biases', [bias.shape for bias in nnet.best_estimator_.intercepts_])
print('*'*40)

y_pred = nnet.predict(X_test)

#%% Save predictions
output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': y_pred})
output.to_csv('predictions.csv', index=False)
print("Submission successfully saved!")

