import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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
y_train = y_train_df.values[:, 0]
print(X_train.shape, X_test.shape, y_train.shape)


# Impute missing values with mean of each collumn
X_mean = np.nanmean(X_train, axis=0, keepdims=True)
X_train = np.where(np.isnan(X_train), X_mean, X_train)
X_test = np.where(np.isnan(X_test), X_mean, X_test)

# select top k features with highest mutual information
k = 100
selection = SelectKBest(mutual_info_regression, k=k).fit(X_train, y_train)
X_train = selection.transform(X_train)
X_test = selection.transform(X_test)



# Calculate the mean and covariance matrix of X_train
mean = np.mean(X_train, axis=0)
cov = np.cov(X_train, rowvar=False)
inv_covmat = inv(cov)

# Calculate Mahalanobis distance for each data point
distances = np.array([np.sqrt((x - mean).T @ inv_covmat @ (x - mean)) for x in X_train])

# Set a threshold for outliers based on a chi-square distribution
# For example, we can set it to the 97.5th percentile
threshold = chi2.ppf(0.90, df=X_train.shape[1])

# Identify the indices of non-outliers
non_outlier_indices = np.where(distances < threshold)[0]

# Filter X_train and y_train to exclude outliers
X_train = X_train[non_outlier_indices]
y_train = y_train[non_outlier_indices]




# Split data into training and validation (80-20)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)


# train Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_train_pred = regressor.predict(X_train)
y_val_pred = regressor.predict(X_val)


# Evaluate model on training and validation sets
train_score = r2_score(y_train, y_train_pred)
val_score = r2_score(y_val, y_val_pred)

print("train score: ", train_score)
print("val score: ", val_score)


# Use model on test data to create output
y_test_pred = regressor.predict(X_test)
table = pd.DataFrame({'id': np.arange(0, y_test_pred.shape[0]), 'y': y_test_pred.flatten()})
table.to_csv("./data/y_test_pred.csv", index=False)
