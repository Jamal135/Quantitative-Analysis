# Creation Date: 19/05/2022

import numpy
import pandas
import seaborn
import matplotlib.pyplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm

dataset = pandas.read_csv("data4.csv")
# orgid,gender,age,cont,neg,jsat,neur,notrain,nofeed
x = dataset[["nofeed", "notrain", "cont", "neur", "age", "neg", "gender"]]
y = dataset["jsat"]

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.3, random_state=100) 

mlr= LinearRegression()
mlr.fit(x_train, y_train)

#Printing the model coefficients
print(mlr.intercept_)
# pair the feature names with the coefficients
print(list(zip(x, mlr.coef_)))
#Predicting the Test and Train set result 
y_pred_mlr= mlr.predict(x_test)
x_pred_mlr= mlr.predict(x_train)
print(f"Prediction for test set: {y_pred_mlr}")
mlr_diff = pandas.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})
print(mlr_diff)

#print(mlr.predict([[4,5,3,2]]))
# print the R-squared value for the model
print('R squared value of the model: {:.2f}'.format(mlr.score(x,y)*100))
# 0 means the model is perfect. Therefore the value should be as close to 0 as possible
meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootMeanSqErr = numpy.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))

print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)
