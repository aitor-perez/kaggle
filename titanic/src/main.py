import numpy as np
import pandas as pd


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
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

y = train_data["Survived"]

model = RandomForestClassifier(random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('predictions.csv', index=False)
print("Your submission was successfully saved!")
