"""
This model uses the World Happiness Report from Kaggle https://www.kaggle.com/unsdsn/world-happiness# to
predict the happiness scores
"""

import numpy as np
import csv
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV

# data for each year
data_2015 = np.array([])
data_2016 = np.array([])
data_2017 = np.array([])

# pre-processing
# columns: 0,3,5,6,7,8,10,9,11
with open('../data/2015.csv', newline='') as csvfile:
    file_2015 = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in file_2015:
        row = np.concatenate((row[0:1], row[3:4], row[5:9], row[10:11], row[9:10], row[11:12]))
        data_2015 = np.append(data_2015, row)

# columns: 0,3,6,7,8,9,11,10,12
with open('../data/2016.csv', newline='') as csvfile:
    file_2016 = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in file_2016:
        row = np.concatenate((row[0:1], row[3:4], row[6:10], row[11:12], row[10:11], row[12:13]))
        data_2016 = np.append(data_2016, row)

# columns: 0,2,5,6,7,8,9,10,11
with open('../data/2017.csv', newline='') as csvfile:
    file_2017 = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in file_2017:
        row = np.concatenate((row[0:1], row[2:3], row[5:12]))
        data_2017 = np.append(data_2017, row)

# extract training data and testing data
data_2015 = data_2015.reshape((-1, 9))
data_2016 = data_2016.reshape((-1, 9))
data_2017 = data_2017.reshape((-1, 9))
train = np.append(data_2015[1:, :], data_2016[1:, :], axis=0)
test = data_2017[1:, :]

train_x = train[:, 2:]
train_y = train[:, 1]
test_x = test[:, 2:]

test_y = test[:, 1]

train_x = train_x.astype(float)
train_y = train_y.astype(float)
test_x = test_x.astype(float)
test_y = test_y.astype(float)

# feature selection
selector = VarianceThreshold()
Trx = selector.fit_transform(train_x, train_y)

# model selection
hyperparameters = {"n_estimators": range(3, 20), "random_state": [0]}
RF = RandomForestRegressor(random_state=0)
clf = GridSearchCV(RF, hyperparameters, "neg_mean_squared_error")
clf.fit(Trx, train_y)
RF = clf.best_estimator_
prediction = RF.predict(test_x)
RMSE = math.sqrt(mean_squared_error(test_y, prediction)/len(test_y))
print(prediction)

print(RMSE)
