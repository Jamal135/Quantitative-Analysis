# Creation Date: 26/05/2022

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import linear_model
from pyprocessmacro import Process
import statsmodels.formula.api as smf
from pingouin import mediation_analysis


def load_data(datafile: str, drop_list: list = None):
    ''' Returns: Loaded Pandas dataframe from CSV file. '''
    if not datafile.endswith(".csv"):
        datafile += ".csv"
    df = pd.read_csv(datafile)
    if drop_list is not None:
        df.drop(drop_list, axis=1, inplace=True)
    return df


def count_column(datafile: str, columns: list):
    ''' Returns: Value count for all selected columns. '''
    df = load_data(datafile)
    for column in columns:
        print(f"{column}:\n{df[column].value_counts()}")


def show_data(datafile, titles: list):
    ''' Purpose: Prints mean, median, mode, and deviation data for selected data columns. '''
    df = load_data(datafile)
    for row in titles:
        print(
            f"{row}: mean {df[row].mean()}, median {df[row].median()}, mode {df[row].mode()}, std {df[row].std()}")


def statsmodel_multiple_regression(datafile: str, dependent: list, independent: list):
    ''' Purpose: Perform linear multiple regression using Statsmodel package. '''
    df = load_data(datafile)
    dependent_data = df(dependent)
    independent_data = df(independent)
    #independent = pd.get_dummies(data=X, drop_first=True)
    regression = linear_model.LinearRegression()
    regression.fit(independent_data, dependent_data)
    print('Intercept: \n', regression.intercept_)
    print('Coefficients: \n', regression.coef_)
    model = sm.OLS(dependent_data, independent_data).fit()
    print(model.predict(independent_data))
    print(model.summary())


def moderated_scatter_plot(df, x_axis: str, y_axis: str, xlabel: str, ylabel: str):
    ''' Purpose: Creates moderation interaction scatter plot. '''
    np.corrcoef(df[x_axis],df[y_axis])
    plt.scatter(df[x_axis],df[y_axis])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    plt.clf()


def statsmodel_moderated_regression(datafile: str, independent: str, moderator: str, 
                                    moderation_formula: str, xlabel: str, ylabel: str):
    ''' Purpose: Performs statsmodel based moderated regression analysis. '''
    df = load_data(datafile)
    df['interaction'] = df[moderator]*df[independent]
    moderated_scatter_plot(df, independent, "interaction", xlabel, ylabel)
    center = lambda x: (x - x.mean())
    df[['moderator_centered','independent_centered']] = df[[moderator,independent]].apply(center)
    df['interaction_centered'] = df['moderator_centered'] * df['independent_centered']
    moderated_scatter_plot(df, independent, "interaction_centered", xlabel, ylabel)
    mod = smf.ols(formula = moderation_formula, data = df).fit()
    print(mod.summary())


def process_mediated_regression(datafile: str, dependent: str, independent: str, mediator: list, controls_list: list = None, controls_argument: str = "all"):
    ''' Purpose: Performs Process Macro based mediated regression analysis. '''
    if controls_list is None:
        controls_list = []
    df = load_data(datafile)
    results = Process(
        data=df,
        model=4,
        x=dependent,
        y=independent,
        m=mediator,
        controls=controls_list,
        controls_in=controls_argument,
        suppr_init=True,
    )
    results.summary()


def pingouin_mediated_regression(datafile: str, dependent: str, independent: str, mediator: str):
    ''' Purpose: Performs mediated regression analysis via the Pingouin package. '''
    df = load_data(datafile)
    results = mediation_analysis(
        data=df,
        x=dependent,
        m=mediator,
        y=independent,
        alpha=0.05,
        seed=42
    )
    print(results)


def binary_logistics_regression(datafile: str, formula: str, category_list: list = None):
    ''' Purpose: Performs binary logistics regression analysis via the Statsmodel package. '''
    df = load_data(datafile)
    if category_list != None:
        for category in category_list:
            df[category] = df[category].astype('category')
    df.info()
    model = smf.logit(formula=formula, data=df).fit()
    print(model.summary())
