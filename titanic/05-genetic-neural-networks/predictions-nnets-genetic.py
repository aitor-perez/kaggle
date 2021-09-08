# Execute after the preprocessing script

import numpy as np
import pandas as pd

pd.options.display.max_columns = 20

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

from sklearn.model_selection import RepeatedKFold

from NNet import NNet

import time

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


#%% Helper function to train nnets
def train_nnets(nnets, X_train, y_train):
    start = time.time()

    # We train the nnets in the population
    print("  ", end="")
    cv = RepeatedKFold(n_splits=10, n_repeats=3)
    for nnet in nnets:
        print("+", end="")
        nnet.train(X_train, y_train, cv)
    print()

    stop = time.time()
    duration = stop - start

    print("  Total training time:", round(duration, 2), "s")

    return nnets


#%% Training of nnet with genetic approach
print("Starting evolution of Neural Networks (generations =", generations, ", population size =", population_size, ")")

hall_of_fame = []
best_scores = []
mean_scores = []
hof_scores = []
for gen in range(generations):
    print("Generation", gen)

    # We train the population of nnets
    nnets = train_nnets(nnets, X_train, y_train)

    # We sort them by score
    nnets.sort(key=lambda nnet:nnet.score, reverse=True)
    best_scores.append(nnets[0].score)
    mean_scores.append(np.mean([nnet.score for nnet in nnets]))

    print("  Sorted nnets:", nnets)

    # We select the best and the worst nnets
    best = nnets[:tail_size]
    worst = nnets[(population_size - tail_size):]

    # We keep track of the all-time best
    hall_of_fame += best
    hall_of_fame.sort(key=lambda nnet:nnet.score, reverse=True)
    hall_of_fame = hall_of_fame[:tail_size]
    hof_scores.append(np.mean([hall_of_famer.score for hall_of_famer in hall_of_fame]))

    # We exit if it is the last generation
    if gen == generations - 1:
        break

    # Nnets survive if they are not among the worst
    survivors = nnets[:(population_size - tail_size)]

    # A fraction of the worst still survives for the sake of diversification
    survivors += list(np.random.choice(worst, exception_size, replace=False))

    # Reproduction
    #   For every child, we select two parents with uneven probabilities,
    #   so that better nnets are selected with higher probability.
    children = []
    for i in range(population_size):
        n = len(survivors)
        probabilities = np.linspace(1.5, 0.5, n) / n
        parents = np.random.choice(survivors, p=probabilities, size=2, replace=False)
        child = parents[0].reproduce(parents[1])
        children.append(child)

    # We update the nnets population for next generation
    nnets = children


#%% We plot the best and the mean score
fig, ax = plt.subplots()
ax.plot(best_scores, label='Generation best')
ax.plot(mean_scores, label='Generation mean')
ax.plot(hof_scores, label='HOF mean')
ax.set_xlabel('Generation')
ax.set_ylabel('Score')
ax.set_title("Score evolution over generations")
ax.legend()
plt.show()

#%% We generate several predictions

# current_best is the nnet with highest score in last generation
current_best = nnets[0]
y_pred_cb = current_best.clf.predict(X_test)

# current_elders are the [tail_size] nnets with highest score in last generation
current_elders = nnets[:tail_size]
y_preds_ce = [current_elder.clf.predict(X_test) for current_elder in current_elders]
y_pred_ce = pd.DataFrame(y_preds_ce).T.mode(axis=1)[0]

# all_time_best is the nnet with highest score through all generations
all_time_best = hall_of_fame[0]
y_pred_atb = all_time_best.clf.predict(X_test)

# all_time_elders are the [tail_size] nnets with highest score through all generations
all_time_elders = hall_of_fame
y_preds_ate = [all_time_elder.clf.predict(X_test) for all_time_elder in all_time_elders]
y_pred_ate = pd.DataFrame(y_preds_ate).T.mode(axis=1)[0]


#%% We save predictions
output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': y_pred_cb})
output.to_csv('predictions/current-best.csv', index=False)

output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': y_pred_ce})
output.to_csv('predictions/current-elders.csv', index=False)

output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': y_pred_atb})
output.to_csv('predictions/all-time-best.csv', index=False)

output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': y_pred_ate})
output.to_csv('predictions/all-time-elders.csv', index=False)

print("Submissions successfully saved!")

