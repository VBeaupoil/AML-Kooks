import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.gaussian_process.kernels import RBF, DotProduct, ExpSineSquared, RationalQuadratic, WhiteKernel, Matern
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, IsolationForest

# import data, leave out header
X_train_df = pd.read_csv('./data/X_train.csv', skiprows=1, header=None)
X_test_df = pd.read_csv('./data/X_test.csv', skiprows=1, header=None)
y_train_df = pd.read_csv('./data/y_train.csv', skiprows=1, header=None)


# remove ID collum (not relevant, equal to index)
X_train = X_train_df.values[:, 1:]
X_test = X_test_df.values[:, 1:]
y_train = y_train_df.values[:, 1]
print(X_train.shape, X_test.shape, y_train.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Impute missing values with median of each collumn
X_m = np.nanmedian(X_train, axis=0, keepdims=True)
X_train = np.where(np.isnan(X_train), X_m, X_train)
X_test = np.where(np.isnan(X_test), X_m, X_test)


#remove outliers using Local Outlier Factor
#results: train score 0.939, val score 0.658
lof = LocalOutlierFactor(n_neighbors=50, contamination=0.1, metric='minkowski', p=1.5 )
outlier_labels = lof.fit_predict(X_train)



#remove outliers with isolation forest
#iso_forest = IsolationForest(contamination=0.1, random_state=10)
#outlier_labels = iso_forest.fit_predict(X_train)

X_train = X_train[outlier_labels == 1]
y_train = y_train[outlier_labels == 1]


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
correlation_threshold = 0.1  # threshold; adjust as needed

# Select features with correlation above the threshold
selected_features = correlation_with_target[abs(correlation_with_target) > correlation_threshold].index
X_train = X_train[selected_features]
X_test = X_test[selected_features]
print(X_train.shape, X_test.shape, y_train.shape)


# select top k features with highest mutual information
k = 150
selection = SelectKBest(mutual_info_regression, k=k).fit(X_train, y_train)
X_train = selection.transform(X_train)
X_test = selection.transform(X_test)

print(X_train.shape, X_test.shape, y_train.shape)


# Split data into training and validation

X_og, y_og = X_train, y_train # maintain copy of full dataset before splitting
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=4)

print(X_train[0,:20])

# train Regression model


estimators = [
    ('rfr', RandomForestRegressor(random_state=0)),
    ('svr', GaussianProcessRegressor(kernel=RationalQuadratic(length_scale=1, alpha=0.01)+Matern(length_scale=1), alpha=0.1, random_state=10)),
    ('svr2', GaussianProcessRegressor(kernel=RBF(length_scale=1)+Matern(), alpha=0.1, random_state=14)),
    ('svr3', GaussianProcessRegressor(kernel=RBF(length_scale=1)+Matern(), alpha=0.1, random_state=10))

]

regressor = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression()
)
#regressor = GaussianProcessRegressor(kernel=RationalQuadratic(length_scale=1, alpha=0.001)+Matern(length_scale=1), alpha=0.1, random_state=10)
#regressor = GaussianProcessRegressor(kernel=RationalQuadratic(length_scale=10, alpha=0.001), alpha=0.1)
regressor.fit(X_train, y_train)
y_train_pred = regressor.predict(X_train)
y_val_pred = regressor.predict(X_val)

# Evaluate model on training and validation sets
train_score = r2_score(y_train, y_train_pred)
val_score = r2_score(y_val, y_val_pred)

print("train score: ", train_score)
print("val score: ", val_score)


#retrain model on whole data since we dont need the validation score anymore
#regressor = GaussianProcessRegressor()
regressor.fit(X_og, y_og)
# Use model on test data to create output
y_test_pred = regressor.predict(X_test)
table = pd.DataFrame({'id': np.arange(0, y_test_pred.shape[0]), 'y': y_test_pred.flatten()})
table.to_csv("./data/y_test_pred.csv", index=False)