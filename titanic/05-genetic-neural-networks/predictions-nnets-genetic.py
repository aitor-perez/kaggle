# Execute after the preprocessing script

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.neural_network import MLPClassifier

pd.options.display.max_columns = 20

#%% Nnet class
class NNet:
    def __init__(self, hls, act, alpha):
        self.hls = hls
        self.act = act
        self.alpha = alpha
        self.score = 0

    def __repr__(self):
        return "MLPClassifier(" + str(self.hls) + ", " + str(self.act) + ", " + str(round(self.alpha, 5)) + "): " + str(round(self.score, 4))

    def __str__(self):
        return "MLPClassifier(" + str(self.hls) + ", " + str(self.act) + ", " + str(round(self.alpha, 5)) + "): " + str(round(self.score, 4))

    def train(self, X, y, cv):
        self.clf = MLPClassifier(max_iter=10000,
                                 hidden_layer_sizes=(self.hls,),
                                 activation=self.act,
                                 alpha=self.alpha)
        self.score = np.mean(cross_val_score(self.clf, X, y, cv=cv, scoring='accuracy'))
        self.clf.fit(X, y)

    def reproduce(self, other):
        return self

    def mutate(self):
        # Mutate hidden_layer_sizes
        inc_hls = round(np.random.uniform(-0.2 * self.hls, 0.2 * self.hls))
        hls = np.maximum(self.hls + inc_hls, 5)

        # Mutate activation function type
        acts = ['logistic', 'tanh', 'relu']
        acts.remove(self.act)
        r = np.random.uniform()
        if r < 0.01:
            act = acts[0]
        elif r > 0.99:
            act = acts[-1]
        else:
            act = self.act

        # Mutate alpha value
        r = np.random.uniform(1, 2)
        alpha = np.random.choice([self.alpha * r, self.alpha / r])

        return NNet(hls, act, alpha)


#%% Read preprocessed data
train_data = pd.read_csv('preprocessed_train.csv')
test_data = pd.read_csv('preprocessed_test.csv')

X_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived']
X_test = test_data.drop('PassengerId', axis=1)
print(X_train.shape, y_train.shape, X_test.shape)

#%% We prepare the initial structures for the genetic approach
generations = 10

population_size = 20
tail_size = round(population_size/4)
exception_size = round(tail_size/2)

initial_params = {
    'hls': list(range(5, 13)),
    'act': ['logistic', 'tanh', 'relu'],
    'alpha': list(np.logspace(-5, -3, 3))
}

nnets = []
for i in range(population_size):
    hls = np.random.choice(initial_params['hls'])
    act = np.random.choice(initial_params['act'])
    alpha = np.random.choice(initial_params['alpha'])
    nnets.append(NNet(hls, act, alpha))

#%% Training of nnet with genetic approach
print("Starting evolution of Neural Networks (generations =", generations, ", population size =", population_size, ")")

history_best = None

for gen in range(generations):
    print("Generation", gen)

    # We train the nnets in the population
    cv = KFold(n_splits=10, shuffle=True)
    n = 0
    for nnet in nnets:
        n += 1
        print("  Training nnet #", n, sep="")
        nnet.train(X_train, y_train, cv)

    # We sort them by score
    nnets.sort(key=lambda nnet:nnet.score, reverse=True)

    print("Sorted nnets", nnets)
    print("Best score", nnets[0].score)
    print("Mean score", np.mean([nnet.score for nnet in nnets]))

    # We keep track of the all-time best
    if history_best is None or nnets[0].score > history_best.score:
        history_best = nnets[0]

    # We exit if it is the last generation
    if gen == generations - 1:
        break

    # We select the best and the worst nnets
    best = nnets[:tail_size]
    worst = nnets[(population_size - tail_size):]

    # Nnets survive if they are not among the worst
    survivors = nnets[:(population_size - tail_size)]

    # A fraction of the worst still survives for the sake of diversification
    survivors += list(np.random.choice(worst, exception_size, replace=False))

    # Reproduction
    children_nnets = []
    # Best nnets always reproduce
    for nnet in best:
        children_nnets.append(nnet.mutate())

    # We complete the children population with descendants of the survivors
    for i in range(population_size - tail_size):
        nnet = np.random.choice(survivors)
        child_nnet = nnet.mutate()
        children_nnets.append(child_nnet)

    # We update the nnets population for next generation
    nnets = children_nnets

#%% We choose the best nnet for prediction
best_nnet = nnets[0]
# best_nnet = history_best

#%% Generate the predictions
y_pred = best_nnet.clf.predict(X_test)

#%% Save predictions
output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': y_pred})
output.to_csv('predictions.csv', index=False)
print("Submission successfully saved!")

