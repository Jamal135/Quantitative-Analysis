# Creation Date: 19/04/2022


import pandas
from factor_analyzer import ConfirmatoryFactorAnalyzer, ModelSpecificationParser


# Multi-Factor CFA


# What variables compose each factor:
model_dict = {"F1": ["V1", "V2", "V3"],
              "F2": ["V5", "V6", "V7"]}


def load_CSV(filename: str, drop_list: list = None):
    ''' Returns: CSV loaded to dataframe with select columns dropped. '''
    dataframe = pandas.read_csv(f"{filename}.csv")
    if drop_list is not None:
        dataframe.drop(drop_list, axis=1, inplace=True)
    return dataframe


def cfa_analysis(filename: str, datafile: str, model_dict: dict):
    ''' Purpose: Complete Confirmatory Factor Analysis. '''
    df = load_CSV(datafile, ["subno"])
    model_spec = ModelSpecificationParser.parse_model_specification_from_dict(df, model_dict)
    cfa = ConfirmatoryFactorAnalyzer(model_spec, disp=False)
    cfa.fit(df.values)
    # measures...
