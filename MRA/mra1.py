import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm


df = pd.read_csv("data5.csv")

X = df[["nofeed", "notrain", "cont", "neur", "neg", "gender", "age", "orgidA", "orgidB", "orgidC"]]
Y = df["jsat"]
 
X = pd.get_dummies(data=X, drop_first=True)

# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# with statsmodels
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)