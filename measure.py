# Creation Date: 26/05/2022

import pandas

def load_CSV(filename: str, drop_list: list = None):
    ''' Returns: CSV loaded to dataframe with select columns dropped. '''
    if drop_list is None:
        drop_list = []
    dataframe = pandas.read_csv(f"{filename}.csv")
    if drop_list is not None:
        dataframe.drop(drop_list, axis=1, inplace=True)
    return dataframe

df = load_CSV("data5")
print(df['gender'].value_counts())
print(df['orgid'].value_counts())
print(df['age'].value_counts())