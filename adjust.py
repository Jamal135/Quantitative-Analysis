# Creation Date: 18/04/2022

import numpy
import pandas

def load_CSV(filename: str, drop_list: list = None):
    ''' Returns: CSV loaded to dataframe with select columns dropped. '''
    if drop_list is None:
        drop_list = []
    dataframe = pandas.read_csv(f"{filename}.csv")
    if drop_list is not None:
        dataframe.drop(drop_list, axis=1, inplace=True)
    return dataframe

df = load_CSV("data2")
filename = "data5"
df["gender"] = numpy.where(df["gender"] == "female", 0, 1)
df["orgidA"] = numpy.where(df["orgid"] == "Org A", 1, 0)
df["orgidB"] = numpy.where(df["orgid"] == "Org B", 1, 0)
df["orgidC"] = numpy.where(df["orgid"] == "Org C", 1, 0)
df.to_csv(f"{filename}.csv", index=False)

measure_list = [["cont1", "cont2", "cont3"], ["neg1", "neg2", "neg3", "neg4", "neg5", "neg6", "neg7", "neg8", "neg9", "neg10", "neg11"], ["jsat1", "jsat2", "jsat3"], ["neur1", "neur2", "neur3"], ["notrain1", "notrain2", "notrain3"], ["nofeed1", "nofeed2", "nofeed3"]]
variable_list = ["cont", "neg", "jsat", "neur", "notrain", "nofeed"]
for index, measures in enumerate(measure_list):
    df[variable_list[index]] = df[measures].sum(axis=1)/len(measures)
    df.drop(measures, axis = 1, inplace=True)
df.to_csv(f"{filename}.csv", index=False)