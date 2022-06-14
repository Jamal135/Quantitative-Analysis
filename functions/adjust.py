# Creation Date: 18/04/2022

import numpy
import pandas


def convert_CSV(filename: str):
    ''' Purpose: Converts SAV to CSV file. '''
    dataframe = pandas.read_spss(f"{filename}.sav")
    dataframe.to_csv(f"{filename}.csv", index=False)


def load_CSV(filename: str, drop_list: list = None):
    ''' Returns: CSV loaded to dataframe with select columns dropped. '''
    if drop_list is None:
        drop_list = []
    dataframe = pandas.read_csv(f"{filename}.csv")
    if drop_list is not None:
        dataframe.drop(drop_list, axis=1, inplace=True)
    return dataframe


def combine_variables(datafile: str, outputfile: str, variable_list: list, factor_list: list):
    ''' Purpose: Combines variables into factor results (average). '''
    df = load_CSV(datafile)
    for index, measures in enumerate(variable_list):
        df[factor_list[index]] = df[measures].sum(axis=1)/len(measures)
        df.drop(measures, axis=1, inplace=True)
    df.to_csv(f"{outputfile}.csv", index=False)


def dummy_code(datafile: str, outputfile: str, result_list: list, factor_list: list):
    ''' Purpose: Creates dummy codes for selected factors given each result. '''
    df = load_CSV(datafile)
    for i, factor in enumerate(factor_list):
        df[factor] = numpy.where(df[factor] == result_list[i], 0, 1)
    df.to_csv(f"{outputfile}.csv", index=False)


def value_count(datafile: str, columns_list: list):
    ''' Purpose: Prints value counts for selected columns. '''
    df = load_CSV(datafile)
    for column in columns_list:
        print(df[column].value_counts())


def data_spread(datafile: str, columns_list: list):
    ''' Purpose: Prints mean, median, mode, and STD for columns. '''
    df = load_CSV(datafile)
    for column in columns_list:
        print(f"{column}:")
        print(df[column].mean())
        print(df[column].median())
        print(df[column].mode())
        print(df[column].std())
        print("\n")


scale = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly agree": 5
}


def map_values(datafile: str, outputfile: str, column_list: list, dictionary: dict):
    ''' Purpose: Map values to integers given dictionary. '''
    df = load_CSV(datafile)
    for column in column_list:
        df.replace({column: dictionary}, inplace=True)
    df.to_csv(f"{outputfile}.csv", index=False)
