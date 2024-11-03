import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor

# import data, leave out header
X_train_df = pd.read_csv('./data/X_train.csv', skiprows=1, header=None)
X_test_df = pd.read_csv('./data/X_test.csv', skiprows=1, header=None)
y_train_df = pd.read_csv('./data/y_train.csv', skiprows=1, header=None)


# remove ID collum (not relevant, equal to index)
X_train = X_train_df.values[:, 1:]
X_test = X_test_df.values[:, 1:]
y_train = y_train_df.values[:, 1]
print(X_train.shape, X_test.shape, y_train.shape)

imputer = IterativeImputer(estimator=DecisionTreeRegressor(), max_iter=5, random_state=42)
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


X_train.to_csv("./data/X_test_imp.csv", index=False)
X_test.to_csv("./data/X_train_imp.csv", index=False)