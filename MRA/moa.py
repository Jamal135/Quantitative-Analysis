
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

df = pd.read_csv("data5.csv")

df['interaction'] = df.cont*df.notrain
np.corrcoef(df.notrain,df.interaction)
plt.scatter(df.notrain,df.interaction)
plt.xlabel('Job Control * Lack of Training')
plt.ylabel('Lack of Training')
plt.show()
center = lambda x: (x - x.mean())
df[['cont_centered','notrain_centered']] = df[['cont','notrain']].apply(center)
df['interaction_centered'] = df['cont_centered'] * df['notrain_centered']
np.corrcoef(df.notrain,df.interaction_centered)
plt.scatter(df.notrain,df.interaction_centered)
plt.xlabel('Job Control * Lack of Training')
plt.ylabel('Lack of Training')
plt.show()
mod = smf.ols(formula = "jsat ~ cont + notrain + interaction_centered + gender + age + orgidA + orgidB + orgidC", data = df).fit()
print(mod.summary())