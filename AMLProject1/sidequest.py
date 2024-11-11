import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.gaussian_process.kernels import RBF, DotProduct, ExpSineSquared, RationalQuadratic, WhiteKernel, Matern
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# import data, leave out header
X_train_df = pd.read_csv('./data/X_train_imp1.csv', skiprows=1, header=None)
X_test_df = pd.read_csv('./data/X_test_imp1.csv', skiprows=1, header=None)
y_train_df = pd.read_csv('./data/y_train.csv', skiprows=1, header=None)


# remove ID collum (not relevant, equal to index)
X_train = X_train_df.values[:, 1:]
X_test = X_test_df.values[:, 1:]
y_train = y_train_df.values[:, 1]
print(X_train.shape, X_test.shape, y_train.shape)


# remove outliers
lof = LocalOutlierFactor(n_neighbors=50, contamination=0.15, metric='minkowski', p=1.5)
outlier_labels = lof.fit_predict(X_train)
X_train = X_train[outlier_labels == 1]
y_train = y_train[outlier_labels == 1]


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


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

"""
# select top k features with highest mutual information
k = 170
selection = SelectKBest(mutual_info_regression, k=k).fit(X_train, y_train)
X_train = selection.transform(X_train)
X_test = selection.transform(X_test)
"""


pca = PCA(n_components=150)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)


X_og, y_og = X_train, y_train # maintain copy of full dataset before splitting
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=5)


#create clusters of the data
clf = KMeans(n_clusters=3, random_state=5, n_init="auto")
clf.fit(X_train)
train_group = clf.predict(X_train)
val_group = clf.predict(X_val)
test_group = clf.predict(X_test)

#create dataframe with cluster asignment
X_train = pd.DataFrame(X_train)
X_val = pd.DataFrame(X_val)
X_test = pd.DataFrame(X_test)
X_train['target'] = y_train
X_train['group'] = train_group
X_val['group'] = val_group
X_val['target'] = y_val
X_test['group'] = test_group

#split data by cluster
grouped_train = {category: group.drop(columns='group') for category, group in X_train.groupby('group')}
grouped_val = {category: group.drop(columns='group') for category, group in X_val.groupby('group')}
grouped_test = {category: group.drop(columns='group') for category, group in X_test.groupby('group')}

print(grouped_val)

regressors = [GaussianProcessRegressor(kernel= Matern(length_scale=100)*Matern(length_scale=100), alpha=0.01),
              GaussianProcessRegressor(kernel=RBF(length_scale=5)+RBF(length_scale=7), alpha=0.01, random_state=50),
              GaussianProcessRegressor(kernel=RBF(length_scale=10)+Matern(), alpha=0.1, random_state=14)]

y_train_pred_full = np.empty((0,))
y_val_pred_full = np.empty((0,))
y_test_pred = np.empty((0,))
# create and apply seperate regressor for each cluster
for category, group_df in grouped_train.items():
    y = group_df['target']
    X = group_df.drop(columns='target')
    yval = grouped_val[category]['target']
    Xval = grouped_val[category].drop(columns='target')
    test = grouped_test[category]

    regressor = regressors[category]
    regressor.fit(X,y)

    y_train_pred = regressor.predict(X)
    y_val_pred = regressor.predict(Xval)

    # Evaluate model on training and validation sets
    train_score = r2_score(y, y_train_pred)
    val_score = r2_score(yval, y_val_pred)
    print(len(y))

    print("train score: ", train_score)
    print("val score: ", val_score)



#retrain model on whole data since we dont need the validation score anymore
#regressor.fit(X_og, y_og)
# Use model on test data to create output


table = pd.DataFrame({'id': np.arange(0, y_test_pred.shape[0]), 'y': y_test_pred.flatten()})
table.to_csv("./data/y_test_pred.csv", index=False)

