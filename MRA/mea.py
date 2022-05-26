# Creation Date: 26/05/2022

import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyprocessmacro import Process
import statsmodels.api as sm
import statsmodels.genmod.families.links as links
from statsmodels.stats.mediation import Mediation

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


import patsy
outcome = np.asarray(df["cong_mesg"])
outcome_exog = patsy.dmatrix("emo + treat + age + educ + gender + income", df,
                             return_type='dataframe')
probit = sm.families.links.probit
outcome_model = sm.GLM(outcome, outcome_exog, family=sm.families.Binomial(link=probit()))
mediator = np.asarray(df["emo"])
mediator_exog = patsy.dmatrix("treat + age + educ + gender + income", df,
                              return_type='dataframe')
mediator_model = sm.OLS(mediator, mediator_exog)
tx_pos = [outcome_exog.columns.tolist().index("treat"),
          mediator_exog.columns.tolist().index("treat")]
med_pos = outcome_exog.columns.tolist().index("emo")
med = Mediation(outcome_model, mediator_model, tx_pos, med_pos).fit()
med.summary()