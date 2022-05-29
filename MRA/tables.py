
from calendar import c
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data5.csv")
plt.rcParams['axes.axisbelow'] = True
def create_plot(df, independent, dependent, xlabel, title):
    plt.scatter(df[independent], df[dependent], color='blue')
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('Job Satisfaction', fontsize=14)
    plt.grid(True)
    plt.savefig(f"MRA/{title}.png")

def whisker_plot(df, rows: list, titles: list):
    figure = plt.figure()
    df2 = df
    for i, row in enumerate(rows):
        df2.rename({row: titles[i]}, axis='columns', inplace=True)
    df2.boxplot(column=titles)
    figure.savefig("MRA/whiskers.png")

def show_data(df, titles: list):
    print(df)
    for row in titles:
        print(f"{row}: mean {df[row].mean()}, median {df[row].median()}, mode {df[row].mode()}, std {df[row].std()}")

def create_interaction_plot(independent, dependent, moderator, xlabel):
    plt.scatter(df[independent], df[dependent], c=df[moderator])
    plt.set_cmap("Blues")
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('Job Satisfaction', fontsize=14)
    plt.grid(True)
    plt.colorbar()
    plt.savefig("MRA/interaction.png")

#create_plot(df, "cont", "jsat", "Job Control", "Job Control VS Job Satisfaction")
#create_plot(df, "nofeed", "jsat", "No Feedback", "No Feedback VS Job Satisfaction")
#create_plot(df, "notrain", "jsat", "No Training", "No Training VS Job Satisfaction")
#create_plot(df, "neur", "jsat", "Neuroticism", "Neuroticism VS Job Satisfaction")
#whisker_plot(df, ["cont", "nofeed", "notrain", "neur"], ["Job Control", "Lack of Feedback", "Lack of Training", "Neuroticism"])
#show_data(df, ["Job Control", "Lack of Feedback", "Lack of Training", "Neuroticism"])
create_interaction_plot("notrain", "jsat", "cont", "Lack of Training")