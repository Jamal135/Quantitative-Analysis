# Creation Date: 19/05/2022

import pandas

def convert_CSV(filename: str):
    ''' Purpose: Converts SAV to CSV file. '''
    dataframe = pandas.read_spss(f"{filename}.sav")
    dataframe.to_csv(f"{filename}.csv", index=False)
    print("Conversion Success")

convert_CSV("data2")