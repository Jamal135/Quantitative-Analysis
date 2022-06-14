# Creation Date: 30/05/2022

import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams['axes.axisbelow'] = True


def load_data(datafile: str, drop_list: list = None):
    ''' Returns: Loaded Pandas dataframe from CSV file. '''
    if not datafile.endswith(".csv"):
        datafile += ".csv"
    df = pd.read_csv(datafile)
    if drop_list is not None:
        df.drop(drop_list, axis=1, inplace=True)
    return df


def create_scatter_plot(datafile: str, x_axis: str, y_axis: str,
                        xlabel: str, ylabel: str, title: str):
    ''' Purpose: Creates scatter plot between two variables. '''
    df = load_data(datafile)
    plt.scatter(df[x_axis], df[y_axis], color='chocolate')
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True)
    plt.savefig(f"{title}.png")
    plt.clf()


def whisker_plot(datafile, rows: list, names: list, title: str):
    ''' Purpose: Creates whisker plots for selected columns. '''
    df = load_data(datafile)
    figure = plt.figure()
    for i, row in enumerate(rows):
        df.rename({row: names[i]}, axis='columns', inplace=True)
    df.boxplot(column=names)
    figure.savefig(f"{title}.png")
    figure.clear()


def interaction_scatter_plot(datafile: str, x_axis: str, y_axis: str, z_axis: str, 
                             xlabel: str, ylabel: str, title: str = None):
    ''' Purpose: Creates scatter plot with colour for third term. '''
    df = load_data(datafile)
    plt.scatter(df[x_axis], df[y_axis], c=df[z_axis])
    plt.set_cmap("Blues")
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True)
    plt.colorbar()
    plt.savefig(f"{title}.png")
    plt.clf()
