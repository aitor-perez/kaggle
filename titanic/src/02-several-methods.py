import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

#%% Reading data
train_data = pd.read_csv("../data/train.csv")
test_data = pd.read_csv("../data/test.csv")

#%% Checking data
# See the list of features
print(train_data.columns)

# Classify features
categorical_features = ['Survived', 'Sex', 'Embarked']
ordinal_features = ['Pclass']
continuous_features = ['Age', 'Fare']
discrete_features = ['SibSp', 'Parch']
mixed_features = ['Ticket', 'Cabin']

# Print summaries of the data
print(train_data.info())    # We see that Age, Cabin and Embarked have NA values
print('_'*40)
print(test_data.info())
print('_'*40)
print(train_data.describe())
print('_'*40)
print(test_data.describe())


