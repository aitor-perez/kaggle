import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

pd.options.display.max_columns = 20

#%% Reading data
train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')

#%% See the list of features
print(train_data.columns)

#%% Classify features

# Categorical features: Survived, Sex, Embarked
# Ordinal features: Pclass
# Continuous features: Age, Fare
# Discrete features: SibSp, Parch
# Mixed features: Ticket, Cabin, Name

#%% Print summaries of the data
print(train_data.info())    # We see that Age, Cabin and Embarked have NA values
print('_'*40)
print(test_data.info())
print('_'*40)
print(train_data.describe(include='all'))
print('_'*40)
print(test_data.describe(include='all'))

# We decide to drop PassengerId and Name from the analysis because all values are unique.
# We decide to drop Ticket as well due to the high unique values.
# We decide to drop Cabin due to its large number of NA values.
# We decide to derive Title from the Name feature.

#%% Check for correlations - Summary


def print_stratified_means(feature):
    stratified_means = train_data[[feature, 'Survived']] \
        .groupby([feature], as_index=False) \
        .mean() \
        .sort_values(by='Survived', ascending=False)

    print(stratified_means)


print_stratified_means('Pclass')
print_stratified_means('Sex')
print_stratified_means('SibSp')
print_stratified_means('Parch')

# We see significant correlation with Pclass and Sex, so we decide to include them into the analysis.
# Not significant for SibSp and Parch, we decide to derive another feature from them.

#%% Check for correlations - Age
# g = sns.FacetGrid(train_data, col='Survived')
# g.map(plt.hist, 'Age', alpha=.5, bins=20)
# plt.show()

# We see that infants had high survival rate. We decide to include Age into the analysis.
# However, we decide to discretize the variable into age groups.

#%% Check for correlations - Age and Pclass
# g = sns.FacetGrid(train_data, col='Survived', row='Pclass', height=2.2, aspect=1.6)
# g.map(plt.hist, 'Age', alpha=.5, bins=20)
# g.add_legend()
# plt.show()

# We see that most passengers with Pclass=1 survived, as well as most infants with
# Pclass=2 or Pclass=3. Also most passengers with Pclass=3 did not survive.
# We make clear the need of adding Pclass to the analysis.

#%% Check for correlations - Embarked, Pclass and Sex
# g = sns.FacetGrid(train_data, row='Embarked', height=2.2, aspect=1.6)
# g.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep', order=[1, 2, 3], hue_order=['female', 'male'])
# g.add_legend()
# plt.show()

# We see that females survived much more often than males.
# Embarked=C seems to correlate with higher survival rates for males.
# We decide to include Sex and Embark to the analysis.

#%% Check for correlations - Embarked, Sex, Fare
# g = sns.FacetGrid(train_data, row='Embarked', col='Survived', height=2.2, aspect=1.6)
# g.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None, order=['male', 'female'])
# g.add_legend()
# plt.show()

# We see that passengers having paid high fares had better survival rate. Embark also seems
# to have correlation, which confirms the need to include this feature in the analysis.
# We also decide to discretize Fare.

#%% Drop Ticket and Cabin features, as we decided not to include them

print("Shapes before dropping", train_data.shape, test_data.shape)

train_data = train_data.drop(['Ticket', 'Cabin'], axis=1)
test_data = test_data.drop(['Ticket', 'Cabin'], axis=1)

print("Shapes before dropping", train_data.shape, test_data.shape)

#%% Create derived feature Title from Name

# Create Title feature
for data in [train_data, test_data]:
    data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

print(pd.crosstab(train_data['Title'], train_data['Sex']))

# Replace rare Titles with 'Other', and map to numbers
for data in [train_data, test_data]:
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')

    data.loc[~data['Title'].isin(['Master', 'Miss', 'Mr', 'Mrs']), 'Title'] = "Other"

print(train_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Other': 5}

for data in [train_data, test_data]:
    data['Title'] = data['Title'].map(title_mapping)
    data['Title'] = data['Title'].fillna(0)

print(train_data.head())

#%% Drop Name and PassengerId features, as we decided not to include them

train_data = train_data.drop(['PassengerId', 'Name'], axis=1)
test_data = test_data.drop(['PassengerId', 'Name'], axis=1)

print(train_data.head())
print(test_data.head())
print(train_data.shape, test_data.shape)

#%% Convert Sex feature to numerical

sex_mapping = {'male': 0, 'female': 1}

for data in [train_data, test_data]:
    data['Sex'] = data['Sex'].map(sex_mapping).astype(int)

print(train_data.head())

#%% Complete missing values

# g = sns.FacetGrid(train_data, row='Pclass', col='Sex', height=2.2, aspect=1.6)
# g.map(plt.hist, 'Age', alpha=.5, bins=20)
# g.add_legend()
# plt.show()

# We fill missing Age values with the median Age of the set of passengers with the same Pclass and Sex
guessed_ages = np.zeros((2, 3))

for data in [train_data, test_data]:
    for s in range(0, 2):
        for c in range(0, 3):
            sex_class_data = data[(data['Sex'] == s) & (data['Pclass'] == c + 1)]['Age'].dropna()

            guessed_age = round(2*sex_class_data.median())/2
            guessed_ages[s, c] = guessed_age

    for s in range(0, 2):
        for c in range(0, 3):
            data.loc[(data.Age.isnull()) & (data.Sex == s) & (data.Pclass == c + 1), 'Age'] = guessed_ages[s, c]

    data['Age'] = data['Age'].astype(int)

print(guessed_ages)
print(train_data.head())

#%% Create derived discrete AgeGroup feature from Age

train_data['AgeGroup'] = pd.cut(train_data['Age'], 5)

print(train_data[['AgeGroup', 'Survived']].groupby(['AgeGroup'], as_index=False).mean().sort_values(by='AgeGroup'))

# Replace with ordinals
for data in [train_data, test_data]:
    data.loc[data['Age'] <= 16, 'Age'] = 0
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
    data.loc[data['Age'] > 64, 'Age'] = 4

# Now we can remove the AgeGroup feature
train_data = train_data.drop('AgeGroup', axis=1)

print(train_data.head())

#%% Create derived feature FamilySize from SibSp and Parch

for data in [train_data, test_data]:
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

print(train_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))

#%% Create derived feature IsAlone from FamilySize

for data in [train_data, test_data]:
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

print(train_data[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean().sort_values(by='Survived', ascending=False))

#%% We drop SibSp, Parch and FamilySize and leave just IsAlone

train_data = train_data.drop(['SibSp', 'Parch', 'FamilySize'], axis=1)
test_data = test_data.drop(['SibSp', 'Parch', 'FamilySize'], axis=1)

print(train_data.head())

#%% We create an artificial feature Age*Class from Age and Pclass

for data in [train_data, test_data]:
    data['Age*Class'] = data['Age'] * data['Pclass']

print(train_data.head())

#%% Complete Embarked with the most common occurrence
most_common_port = train_data['Embarked'].dropna().mode()[0]

for data in [train_data, test_data]:
    data.loc[(data['Embarked'].isnull()), 'Embarked'] = most_common_port

print(train_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))

#%% Convert Embarked to numeric

port_mapping = {'S': 0, 'C': 1, 'Q': 2}

for data in [train_data, test_data]:
    data['Embarked'] = data['Embarked'].map(port_mapping).astype(int)

print(train_data.head())

#%% Complete missing Fare from test dataset
test_data['Fare'].fillna(test_data['Fare'].dropna().median(), inplace=True)

print(train_data.info())

#%% Create derived feature FareGroup from Fare
train_data['FareGroup'] = pd.qcut(train_data['Fare'], 4)

print(train_data[['FareGroup', 'Survived']].groupby(['FareGroup'], as_index=False).mean().sort_values(by='Survived', ascending=True))

# Replace with ordinals
for data in [train_data, test_data]:
    data.loc[data['Fare'] <= 7.91, 'Fare'] = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare'] = 2
    data.loc[data['Fare'] > 31, 'Fare'] = 3
    data['Fare'] = data['Fare'].astype(int)

# Now we can remove the FareGroup feature
train_data = train_data.drop('FareGroup', axis=1)

print(train_data.head())
print(test_data.head())

#%% We now have clean preprocessed datasets ready to be used to train models
train_data.to_csv('preprocessed_train.csv', index=False)
test_data.to_csv('preprocessed_test.csv', index=False)




