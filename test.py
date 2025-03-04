import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.set_printoptions(suppress=True)

dataset = pd.read_csv("Dataset/ml.csv",usecols=['tds','turbidty','ph','conductivity','temperature'])
dataset.fillna(0, inplace = True)
Y = dataset.values[:,1:2]
Y = Y.reshape(Y.shape[0],1)
dataset.drop(['turbidty'], axis = 1,inplace=True)
X = dataset.values

sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)
Y = sc.fit_transform(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

#training with random forest
rf_regression = RandomForestRegressor()
rf_regression.fit(X_train, y_train.ravel())
predict = rf_regression.predict(X_test)
predict = predict.reshape(predict.shape[0],1)
predict = sc.inverse_transform(predict)
predict = predict.ravel()
labels = sc.inverse_transform(y_test)
labels = labels.ravel()
print("Predicted Growth: "+str(predict))
print("\nOriginal Growth: "+str(labels))


plt.plot(labels, color = 'red', label = 'Original Growth')
plt.plot(predict, color = 'green', label = 'SVR Predicted Growth')
plt.title('Random Forest Regression Banana Growth Forecasting')
plt.xlabel('Test Data')
plt.ylabel('Forecasting Growth')
plt.legend()
plt.show()

