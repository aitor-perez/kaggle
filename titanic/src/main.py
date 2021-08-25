import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

#%% Reading data
train_data = pd.read_csv("../data/train.csv")
test_data = pd.read_csv("../data/test.csv")

#%% First guess: women survived while men did not
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of women who survived:", rate_women)   # ~0.74
print("% of men who survived:", rate_men)       # ~0.19


#%% Improving prediction with a random forest
features = ["Pclass", "Sex", "SibSp", "Parch"]
X_train = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

y_train = train_data["Survived"]

#%% Cross-validation
cv = KFold(n_splits=10, shuffle=True, random_state=0)

clfs = []
scores = []
for n in [10, 50, 100]:
    print("n = ", n)
    clf = RandomForestClassifier(n_estimators=n, max_depth=5, random_state=0)
    score = np.mean(cross_val_score(clf, X_train, y_train, cv=cv, n_jobs=1, scoring='accuracy'))

    clfs.append(clf)
    scores.append(score)


clf = clfs[np.array(scores).argmax()]
print("Best classifier:", clf)

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('predictions.csv', index=False)
print("Submission successfully saved!")
