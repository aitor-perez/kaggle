import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

pd.options.display.max_columns = 20

#%% Reading data
train_data = pd.read_csv("../data/train.csv")
test_data = pd.read_csv("../data/test.csv")

#%% See the list of features
print(train_data.columns)

#%% Classify features
categorical_features = ['Survived', 'Sex', 'Embarked']
ordinal_features = ['Pclass']
continuous_features = ['Age', 'Fare']
discrete_features = ['SibSp', 'Parch']
mixed_features = ['Ticket', 'Cabin']

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
g = sns.FacetGrid(train_data, col='Survived')
g.map(plt.hist, 'Age', alpha=.5, bins=20)
plt.show()

# We see that infants had high survival rate. We decide to include Age into the analysis.
# However, we decide to discretize the variable into age groups.

#%% Check for correlations - Age and Pclass
g = sns.FacetGrid(train_data, col='Survived', row='Pclass', height=2.2, aspect=1.6)
g.map(plt.hist, 'Age', alpha=.5, bins=20)
g.add_legend()
plt.show()

# We see that most passengers with Pclass=1 survived, as well as most infants with
# Pclass=2 or Pclass=3. Also most passengers with Pclass=3 did not survive.
# We make clear the need of adding Pclass to the analysis.

#%% Check for correlations - Embarked, Pclass and Sex
g = sns.FacetGrid(train_data, row='Embarked', height=2.2, aspect=1.6)
g.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep', order=[1, 2, 3], hue_order=['female', 'male'])
g.add_legend()
plt.show()

# We see that females survived much more often than males.
# Embarked=C seems to correlate with higher survival rates for males.
# We decide to include Sex and Embark to the analysis.

#%% Check for correlations - Embarked, Sex, Fare
g = sns.FacetGrid(train_data, row='Embarked', col='Survived', height=2.2, aspect=1.6)
g.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None, order=['male', 'female'])
g.add_legend()
plt.show()

# We see that passengers having paid high fares had better survival rate. Embark also seems
# to have correlation, which confirms the need to include this feature in the analysis.
# We also decide to discretize Fare.
