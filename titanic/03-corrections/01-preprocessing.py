import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

pd.options.display.max_columns = 20

#%% Read data
train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')

#%% See the list of features
print(train_data.columns)

#%% Classify features

# Numerical:
#   Discrete: SibSp, Parch
#   Continuous: Age, Fare
# Categorical: Survived, Sex, Embarked
# Ordinal: PassengerId, Pclass
# Mixed: Name, Ticket, Cabin

#%% Print NA info of the data
print(train_data.info())
print(test_data.info())

#%% Cabin has many NA values, we drop it
train_data = train_data.drop(['Cabin'], axis=1)
test_data = test_data.drop(['Cabin'], axis=1)

#%% Print summaries of the data
print(train_data.describe(include='all'))
print(test_data.describe(include='all'))

#%% We drop PassengerId from the training set because all values are unique
train_data = train_data.drop(['PassengerId'], axis=1)

#%% We drop Ticket because most values are unique
train_data = train_data.drop(['Ticket'], axis=1)
test_data = test_data.drop(['Ticket'], axis=1)


#%% We use Name to derive a new feature Title, then drop Name
for data in [train_data, test_data]:
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    print(pd.crosstab(data['Title'], data['Sex']))

train_data = train_data.drop(['Name'], axis=1)
test_data = test_data.drop(['Name'], axis=1)

#%% We update rare Titles with more common ones (Master, Miss, Mr, Mrs)
for data in [train_data, test_data]:
    for title in ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir']:
        data['Title'] = data['Title'].replace(title, 'Mr')
    for title in ['Countess', 'Lady', 'Mme', 'Ms', 'Dona']:
        data['Title'] = data['Title'].replace(title, 'Mrs')
    for title in ['Mlle']:
        data['Title'] = data['Title'].replace(title, 'Miss')
    for title in ['Dr']:
        data.loc[(data['Title'] == title) & (data['Sex'] == 'female'), 'Title'] = 'Mrs'
        data.loc[(data['Title'] == title) & (data['Sex'] == 'male'), 'Title'] = 'Mr'

print(train_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# Idea: Perhaps Sex is redundant having Title.
# As it turns out, all females have Mrs or Miss as Title and all males either Mr or Master
# Worth checking: Drop Sex?

#%% We explore the relation between Pclass and Fare
fig, ax = plt.subplots()
third_class_fares = train_data.loc[train_data['Pclass'] == 3, 'Fare'].sort_values()
second_class_fares = train_data.loc[train_data['Pclass'] == 2, 'Fare'].sort_values()
first_class_fares = train_data.loc[train_data['Pclass'] == 1, 'Fare'].sort_values()

ax.plot(np.linspace(0, 1, len(third_class_fares)), third_class_fares)
ax.plot(np.linspace(0, 1, len(second_class_fares)), second_class_fares)
ax.plot(np.linspace(0, 1, len(first_class_fares)), first_class_fares)
#plt.show()

# Idea: Perhaps Fare is redundant having Pclass. It might even be misleading:
# some important people who likely survived might not have paid any fare,
# but stayed in first class.
# Worth checking: Drop Fare?

#%% We fill missing Age values with the median Age of the set of passengers with the same Pclass and Sex
sex_levels = np.sort(train_data['Sex'].unique())
class_levels = np.sort(train_data['Pclass'].unique())

guessed_ages = pd.DataFrame(index=sex_levels, columns=class_levels, data=0)

for data in [train_data, test_data]:
    for s in sex_levels:
        for c in class_levels:
            sex_class_data = data[(data['Sex'] == s) & (data['Pclass'] == c)]['Age'].dropna()

            guessed_age = round(sex_class_data.median(), 2)
            guessed_ages.loc[s, c] = guessed_age

    for s in sex_levels:
        for c in class_levels:
            data.loc[(data.Age.isnull()) & (data.Sex == s) & (data.Pclass == c), 'Age'] = guessed_ages.loc[s, c]

print(train_data.info())
print(test_data.info())

#%% Complete Embarked with the most common occurrence
most_common_port = train_data['Embarked'].dropna().mode()[0]
train_data['Embarked'].fillna(most_common_port, inplace=True)

print(train_data.info())

#%% Complete Fare with the median value
median_fare = test_data['Fare'].dropna().median()
test_data['Fare'].fillna(median_fare, inplace=True)

print(test_data.info())

#%% We convert categorical variable Embarked with one-hot encoding
train_data = pd.get_dummies(train_data, columns=['Embarked'], prefix='Embarked')
test_data = pd.get_dummies(test_data, columns=['Embarked'], prefix='Embarked')

#%% We convert categorical variable Title with one-hot encoding
train_data = pd.get_dummies(train_data, columns=['Title'], prefix='Title')
test_data = pd.get_dummies(test_data, columns=['Title'], prefix='Title')

#%% We convert categorical binary variable Sex to numerical (0=female, 1=male)
for data in [train_data, test_data]:
    for i in range(0, len(sex_levels)):
        data['Sex'] = data['Sex'].replace(sex_levels[i], i)

#%% We normalize all variables to [0, 1]
for data in [train_data, test_data]:
    for feature in ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']:
        min_value = data[feature].min()
        max_value = data[feature].max()
        data[feature] = (data[feature] - min_value) / (max_value - min_value)

print(train_data)
print(test_data)


#%% We now have clean preprocessed datasets ready to be used to train models
train_data.to_csv('preprocessed_train.csv', index=False)
test_data.to_csv('preprocessed_test.csv', index=False)
print("Preprocessed datasets successfully saved!")
