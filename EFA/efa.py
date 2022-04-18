# Creation Date: 18/04/2022


import itertools
import numpy
import pandas
import matplotlib
import matplotlib.pyplot
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo


# Matplot setup
matplotlib.rcParams['figure.figsize'] = (10.0, 6.0)
matplotlib.use('Agg')


def base26(n):
    ''' Returns: Base26 representation of integer. '''
    a = ''
    while n:
        m = n % 26
        if m > 0:
            n //= 26
            a += chr(64 + m)
        else:
            n = n // 26 - 1
            a += 'Z'
    return a[::-1]


def load_CSV(filename: str, drop_list: list = None):
    ''' Returns: CSV loaded to dataframe with select columns dropped. '''
    dataframe = pandas.read_csv(f"{filename}.csv")
    if drop_list is not None:
        dataframe.drop(drop_list, axis=1, inplace=True)
    return dataframe


def adequacy_test(df):
    ''' Purpose: Measures adequacy of data for factorability. '''
    chi_square_score, p_value = calculate_bartlett_sphericity(df)  # Bartlett's Test
    _, kmo_score = calculate_kmo(df)  # Kaiser-Meyer-Olkin Test
    print(f"Bartlett's Test: {(chi_square_score, p_value)}")
    print(f"Kaiser-Meyer-Olkin Test: {(kmo_score)}")


def plot_eisen_values(df, filename: str, eigen_values: list, save_png: bool = True):
    ''' Purpose: Creates visualisation of eisen values. '''
    matplotlib.pyplot.scatter(range(1, df.shape[1]+1), eigen_values)
    matplotlib.pyplot.plot(range(1, df.shape[1]+1), eigen_values)
    matplotlib.pyplot.title('Eigen Value Factor Test')
    matplotlib.pyplot.xlabel('Factors')
    matplotlib.pyplot.ylabel('Eigen Value')
    matplotlib.pyplot.axhline(y=1, c='k')
    if save_png:
        matplotlib.pyplot.savefig(f"EFA/{filename}_Eisen_Figure.png")
    matplotlib.pyplot.show()


def plot_efa_results(efa, filename: str, number_topics: int, headers: list,
                     save_png: bool = True):
    ''' Purpose: Creates visualisation of EFA results. '''
    GRID = numpy.abs(efa.loadings_)
    xlabels = [f"Factor {base26(x + 1)}" for x in range(number_topics)]
    figure, axis = matplotlib.pyplot.subplots()
    x = axis.pcolor(GRID, cmap="Blues")
    figure.colorbar(x, ax=axis)
    axis.set_yticks(numpy.arange(efa.loadings_.shape[0])+0.5, minor=False)
    axis.set_xticks(numpy.arange(efa.loadings_.shape[1])+0.5, minor=False)
    axis.set_yticklabels(headers)
    axis.set_xticklabels(xlabels)
    figure.tight_layout()
    if save_png:
        matplotlib.pyplot.savefig(f"EFA/{filename}_EFA_Figure.png")
    matplotlib.pyplot.show()


def determine_topics(filename: str, datafile: str):
    ''' Purpose: Analysis results to determine topic number. '''
    df = load_CSV(datafile, ["subno"])
    adequacy_test(df)
    gfa = FactorAnalyzer()
    gfa.fit(df)
    eigen_values, _ = gfa.get_eigenvalues()
    plot_eisen_values(df, filename, eigen_values)


# Rotation methods: https://factor-analyzer.readthedocs.io/en/latest/
def EFA_analysis(filename: str, datafile: str, number_topics: int, method: str = "varimax"):
    ''' Purpose: Complete Exploratory Factor Analysis. '''
    df = load_CSV(datafile, ["subno"])
    headers = list(df.columns.values)
    efa = FactorAnalyzer()
    efa.set_params(n_factors=number_topics, rotation=method)
    efa.fit(df)
    plot_efa_results(efa, filename, number_topics, headers)
    summary = efa.get_factor_variance()
    print(f"Correlation Scores: \n{efa.loadings_}")
    print(f"Variance: {summary[0]}")
    print(f"Proportional Variance: {summary[1]}")
    print(f"Cumulative Variance: {summary[2]}")


determine_topics("Assessment", "Data")
#EFA_analysis("Assessment", "Data", 3)


topic_options = [2, 3]
rotation_options = ["varimax", "oblimin", "quartimax", "equamax", "promax"]
def EFA_pipeline(topic_options: list, rotation_options: list):
    for topics, rotation in itertools.product(topic_options, rotation_options):
        EFA_analysis(f"EFA_{topics}_{rotation}_Figure",
                     "Data", topics, method=rotation)
