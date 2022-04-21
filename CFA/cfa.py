# Creation Date: 19/04/2022


import numpy
import pandas
import semopy
from factor_analyzer import ConfirmatoryFactorAnalyzer, ModelSpecificationParser


# Multi-Factor CFA


# What variables compose each factor:
model_dict = {"F1": ["V1", "V2", "V3"],
              "F2": ["V5", "V6", "V7"]}


def load_CSV(filename: str, drop_list: list):
    ''' Returns: CSV loaded to dataframe with select columns dropped. '''
    dataframe = pandas.read_csv(f"{filename}.csv")
    if drop_list is not None:
        dataframe.drop(drop_list, axis=1, inplace=True)
    return dataframe


def cfa_analysis(datafile: str, model_dict: dict, drop_list: list = None):
    ''' Purpose: Complete Confirmatory Factor Analysis. '''
    df = load_CSV(datafile, drop_list)
    print(df)
    model_spec = ModelSpecificationParser.parse_model_specification_from_dict(
        df, model_dict)
    cfa = ConfirmatoryFactorAnalyzer(model_spec, disp=False)
    cfa.fit(df.values)
    print(cfa.loadings_)
    mod = """
                Verbal_IQ =~ Short_Term_Memory + Vocabulary + Word_Processing_Speed + Numerical_Reasoning + General_Knowledge
                Performance_IQ =~ Idea_Generation + Multitasking + Verb_Fluidity
          """
    from semopy import gather_statistics
    model = semopy.Model(mod)
    print(semopy.efa.explore_cfa_model(df))
    model.fit(df)
    cov = model.inspect('mx')['Psi']
    stds = numpy.diagonal(cov) ** (-0.5)
    corr = stds * cov * stds
    print(corr)
    stats = gather_statistics(model)
    print(stats)
    # measures...


model_dict = {"Performance_IQ": ["Idea_Generation", "Multitasking", "Verb_Fluidity"],
              "Verbal_IQ": ["Short_Term_Memory", "Vocabulary", "Word_Processing_Speed", "Numerical_Reasoning", "General_Knowledge"]}
cfa_analysis("data", model_dict, ["subno", "Number_Categorisation", "Workplace_Learning", "Word_Understanding"])