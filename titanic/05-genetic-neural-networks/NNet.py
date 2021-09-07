#%% NNet class to store parameters and classifiers, also
# allowing to reproduce and mutate for genetic selection

import numpy as np

from sklearn.model_selection import cross_val_score

from sklearn.neural_network import MLPClassifier


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

