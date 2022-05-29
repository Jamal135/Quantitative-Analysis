# Creation Date: 26/05/2022

import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyprocessmacro import Process
import statsmodels.api as sm
import statsmodels.genmod.families.links as links
from statsmodels.stats.mediation import Mediation
import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=DeprecationWarning)

df = pd.read_csv("data5.csv")
p = Process(
    data=df,
    model=4,
    x="nofeed",
    y="jsat",
    m=["neur"],
    controls=["age", "gender", "orgidA", "orgidB", "orgidC"],
    controls_in="all",
    suppr_init=True,
)
p.summary()

df = pd.read_csv("data5.csv")

X = df[["neur", "gender", "age", "orgidA", "orgidB", "orgidC"]]
Y = df["nofeed"]
 
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

X = df[["nofeed", "gender", "age", "orgidA", "orgidB", "orgidC"]]
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

import patsy
outcome = np.asarray(df["jsat"])
outcome_exog = patsy.dmatrix("neur + nofeed + jsat", df,
                             return_type='dataframe')
probit = sm.families.links.probit
outcome_model = sm.GLM(outcome, outcome_exog, family=sm.families.Binomial(link=probit()))
mediator = np.asarray(df["neur"])
mediator_exog = patsy.dmatrix("nofeed + jsat", df,
                              return_type='dataframe')
mediator_model = sm.OLS(mediator, mediator_exog)
tx_pos = [outcome_exog.columns.tolist().index("nofeed"),
          mediator_exog.columns.tolist().index("nofeed")]
med_pos = outcome_exog.columns.tolist().index("neur")
med = Mediation(outcome_model, mediator_model, tx_pos, med_pos).fit()
print(med.summary())

from pingouin import mediation_analysis, read_dataset
w = mediation_analysis(data=df, x='nofeed', m='neur', y='jsat', alpha=0.05,
                   seed=42)
print(w)