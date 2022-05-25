# Creation Date: 18/04/2022

import numpy
import pandas

def load_CSV(filename: str, drop_list: list = []):
    ''' Returns: CSV loaded to dataframe with select columns dropped. '''
    dataframe = pandas.read_csv(f"{filename}.csv")
    if drop_list is not None:
        dataframe.drop(drop_list, axis=1, inplace=True)
    return dataframe

df = load_CSV("data3")
filename = "data4"
#df["gender"] = numpy.where(df["gender"] == "female", 0, 1)
#df["orgid"] = numpy.where(df["orgid"] == "Org A", 0, numpy.where(df["orgid"] == "Org B", 1, 2))
#df.to_csv(f"{filename}.csv", index=False)

measure_list = [["cont1", "cont2", "cont3"], ["neg1", "neg2", "neg3", "neg4", "neg5", "neg6", "neg7", "neg8", "neg9", "neg10", "neg11"], ["jsat1", "jsat2", "jsat3"], ["neur1", "neur2", "neur3"], ["notrain1", "notrain2", "notrain3"], ["nofeed1", "nofeed2", "nofeed3"]]
variable_list = ["cont", "neg", "jsat", "neur", "notrain", "nofeed"]
for index, measures in enumerate(measure_list):
    df[variable_list[index]] = df[measures].sum(axis=1)/len(measures)
    df.drop(measures, axis = 1, inplace=True)
df.to_csv(f"{filename}.csv", index=False)