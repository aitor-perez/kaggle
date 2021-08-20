import numpy as np
import pandas as pd

# Reading data
train_data = pd.read_csv("../data/train.csv")
test_data = pd.read_csv("../data/test.csv")

# First guess: women survived while men did not
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of women who survived:", rate_women)   # ~0.74
print("% of men who survived:", rate_men)       # ~0.19

