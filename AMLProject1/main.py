import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy.stats import chi2
from numpy.linalg import inv

# import data, leave out header
X_train_df = pd.read_csv('./data/X_train.csv', skiprows=1, header=None)
X_test_df = pd.read_csv('./data/X_test.csv', skiprows=1, header=None)
y_train_df = pd.read_csv('./data/y_train.csv', skiprows=1, header=None)


# remove ID collum (not relevant, equal to index)
X_train = X_train_df.values[:, 1:]
X_test = X_test_df.values[:, 1:]
y_train = y_train_df.values[:, 1]
print(X_train.shape, X_test.shape, y_train.shape)


# Impute missing values with mean of each collumn
X_m = np.nanmedian(X_train, axis=0, keepdims=True)
X_train = np.where(np.isnan(X_train), X_m, X_train)
X_test = np.where(np.isnan(X_test), X_m, X_test)


#remove outliers
Q1 = np.percentile(X_train, 20, axis=0)
Q3 = np.percentile(X_train, 80, axis=0)
IQR = Q3 - Q1

# Define the outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out rows with outliers
is_not_outlier = np.all((X_train >= lower_bound) & (X_train <= upper_bound), axis=1)
X_train = X_train[is_not_outlier]
y_train = y_train[is_not_outlier]

print(X_train.shape, X_test.shape, y_train.shape)


#find features correlated to output

# Concatenate X_train with y_train to compute correlations
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
data = X_train.copy()
data['target'] = y_train  # Temporarily add y_train as a new column to calculate correlations

# Calculate correlation matrix
correlation_matrix = data.corr()

# Extract correlation of each feature with y_train (target)
correlation_with_target = correlation_matrix['target'].drop('target')  # Drop target itself

# Set a threshold for feature selection
correlation_threshold = 0.03  # Example threshold; adjust as needed

# Select features with correlation above the threshold
selected_features = correlation_with_target[abs(correlation_with_target) > correlation_threshold].index
X_train = X_train[selected_features]
X_test = X_test[selected_features]
print(X_train.shape, X_test.shape, y_train.shape)


# select top k features with highest mutual information
k = 300
selection = SelectKBest(mutual_info_regression, k=k).fit(X_train, y_train)
X_train = selection.transform(X_train)
X_test = selection.transform(X_test)

print(X_train.shape, X_test.shape, y_train.shape)

X_og, y_og = X_train, y_train

# Split data into training and validation (80-20)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=0)


# train Linear Regression model
regressor = RandomForestRegressor(random_state=0)
regressor.fit(X_train, y_train)
y_train_pred = regressor.predict(X_train)
y_val_pred = regressor.predict(X_val)

# Evaluate model on training and validation sets
train_score = r2_score(y_train, y_train_pred)
val_score = r2_score(y_val, y_val_pred)

print("train score: ", train_score)
print("val score: ", val_score)


#regressor = RandomForestRegressor()
#regressor.fit(X_og, y_og)
# Use model on test data to create output
y_test_pred = regressor.predict(X_test)
table = pd.DataFrame({'id': np.arange(0, y_test_pred.shape[0]), 'y': y_test_pred.flatten()})
table.to_csv("./data/y_test_pred.csv", index=False)
