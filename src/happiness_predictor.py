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
import tensorflow as tf
from sklearn.model_selection import train_test_split

# data for each year
# used to store data
data_2015 = np.array([])
data_2016 = np.array([])
data_2017 = np.array([])

# pre-processing
# columns: 0,3,5,6,7,8,10,9,11
# feature: [country, score, economy, family, health, freedom, generosity, trust(Government), Dystopia residual]
with open('../data/2015.csv', newline='') as csvfile:
    file_2015 = csv.reader(csvfile, delimiter=',', quotechar='|')  # read entire file
    for row in file_2015:  # get data row by row
        row = np.concatenate((row[0:1], row[3:4], row[5:9], row[10:11], row[9:10],
                              row[11:12]))  # combine data base on corresponding index
        data_2015 = np.append(data_2015, row)  # add current row into data_2015

# columns: 0,3,6,7,8,9,11,10,12
# feature: [country, score, economy, family, health, freedom, generosity, trust(Government), Dystopia residual]
with open('../data/2016.csv', newline='') as csvfile:
    file_2016 = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in file_2016:
        row = np.concatenate((row[0:1], row[3:4], row[6:10], row[11:12], row[10:11], row[12:13]))
        data_2016 = np.append(data_2016, row)

# columns: 0,2,5,6,7,8,9,10,11
# feature: [country, score, economy, family, health, freedom, generosity, trust(Government), Dystopia residual]
with open('../data/2017.csv', newline='') as csvfile:
    file_2017 = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in file_2017:
        row = np.concatenate((row[0:1], row[2:3], row[5:12]))
        data_2017 = np.append(data_2017, row)

# extract training data and testing data
data_2015 = data_2015.reshape((-1, 9))  # [US,2,3,Canada,5,6,China,8,9] -> [ [US,2,3],
                                        #                                    [Canada,5,6],
                                        #                                    [China,8,9] ]
data_2016 = data_2016.reshape((-1, 9))
data_2017 = data_2017.reshape((-1, 9))

big_data = np.concatenate((data_2015[1:, :], data_2016[1:, :], data_2017[1:, :]), axis=0)  # combine all data into one dataset

np.random.shuffle(big_data)  # shuffle data
X_train, X_test, y_train, y_test = train_test_split(big_data[:, 2:], big_data[:, 1], test_size=0.2, random_state=0)  # 80% of data as training data, 20% of data as testing data

X_train = X_train.astype(np.float32)  # change data type to float
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# ======================== Scikit learn model start ===============================
# feature selection
selector = VarianceThreshold()
Trx = selector.fit_transform(X_train, y_train)

# model selection
hyperparameters = {"n_estimators": range(3, 20), "random_state": [0]}
RF = RandomForestRegressor(random_state=0)
clf = GridSearchCV(RF, hyperparameters, "neg_mean_squared_error")
clf.fit(Trx, y_train)
RF = clf.best_estimator_
prediction = RF.predict(X_test)
RMSE = math.sqrt(mean_squared_error(y_test, prediction))

# print out result
print("(scikit-learn)Predict result \n", prediction)
print("(scikit-learn)RMSE: ", RMSE)
# ======================== Scikit learn model end ================================


# ======================== tensorflow model start ================================
# input data and label
tf_x = tf.placeholder(tf.float32, [None, 7])
tf_y = tf.placeholder(tf.float32, [None, 1])

# neural network layers
l1 = tf.layers.dense(tf_x, 10, tf.nn.relu)
output = tf.layers.dense(l1, 1)

# optimize loss
loss = tf.losses.mean_squared_error(tf_y, output)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        _, l, pred = sess.run([train_op, loss, output], feed_dict={tf_x: X_train, tf_y: y_train.reshape(-1, 1)})
    print("(Tensorflow)This are predictions \n", sess.run(output, feed_dict={tf_x: X_test}))
    print("(Tensorflow)This are true labels \n", y_test)
    print("(Tensorflow)RMSE: ", math.sqrt(sess.run(loss, feed_dict={tf_x: X_test, tf_y: y_test.reshape(-1, 1)})))
# ======================== tensorflow model end ================================
