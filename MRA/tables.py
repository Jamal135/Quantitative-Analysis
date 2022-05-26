
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data5.csv")
plt.rcParams['axes.axisbelow'] = True
def create_plot(independent, dependent, xlabel, title):
    plt.scatter(df[independent], df[dependent], color='blue')
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('Job Satisfaction', fontsize=14)
    plt.grid(True)
    plt.savefig(f"MRA/{title}.png")

create_plot("cont", "jsat", "Job Control", "Job Control VS Job Satisfaction")
create_plot("nofeed", "jsat", "No Feedback", "No Feedback VS Job Satisfaction")
create_plot("notrain", "jsat", "No Training", "No Training VS Job Satisfaction")
create_plot("neur", "jsat", "Neuroticism", "Neuroticism VS Job Satisfaction")
