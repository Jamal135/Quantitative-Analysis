# Creation Date: 18/04/2022


import numpy
import pandas
import matplotlib
import matplotlib.pyplot
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo


# Matplot setup
matplotlib.rcParams['figure.figsize'] = (10.0, 6.0)


def load_CSV(filename: str, drop_list: list = None):
    ''' Returns: CSV loaded to dataframe with select columns dropped. '''
    dataframe = pandas.read_csv(f"{filename}.csv")
    if drop_list is not None:
        dataframe.drop(drop_list, axis=1, inplace=True)
    return dataframe


def adequacy_test(df, show_scores: bool = True):
    ''' Purpose: Measures adequacy of data for factorability. '''
    bart_score, p_value = calculate_bartlett_sphericity(df)  # Bartlett's Test
    _, kmo_score = calculate_kmo(df)  # Kaiser-Meyer-Olkin Test
    if show_scores:
        print(f"Bartlett's Test: {(bart_score, p_value)}")
        print(f"Kaiser-Meyer-Olkin Test: {(kmo_score)}")


def plot_eisen_values(df, eigen_values: list, save_png: bool = True):
    ''' Purpose: Creates visualisation of eisen values. '''
    matplotlib.pyplot.scatter(range(1, df.shape[1]+1), eigen_values)
    matplotlib.pyplot.plot(range(1, df.shape[1]+1), eigen_values)
    matplotlib.pyplot.title('Eigen Value Factor Test')
    matplotlib.pyplot.xlabel('Factors')
    matplotlib.pyplot.ylabel('Eigen Value')
    matplotlib.pyplot.axhline(y=1, c='k')
    if save_png:
        matplotlib.pyplot.savefig("test.png")
    matplotlib.pyplot.show()


def plot_efa_results(efa, headers: list, save_png: bool = True):
    ''' Purpose: Creates visualisation of EFA results. '''
    GRID = numpy.abs(efa.loadings_)
    figure, axis = matplotlib.pyplot.subplots()
    x = axis.pcolor(GRID, cmap="Blues")
    figure.colorbar(x, ax=axis)
    axis.set_yticks(numpy.arange(efa.loadings_.shape[0])+0.5, minor=False)
    axis.set_xticks(numpy.arange(efa.loadings_.shape[1])+0.5, minor=False)
    axis.set_yticklabels(headers)
    axis.set_xticklabels(["a", "b"])
    figure.tight_layout()
    if save_png:
        matplotlib.pyplot.savefig("test1.png")
    matplotlib.pyplot.show()


def determine_topics():
    ''' Purpose: Analysis results to determine topic number. '''
    df = load_CSV("data", ["subno"])
    adequacy_test(df, show_scores = True)
    gfa = FactorAnalyzer()
    gfa.fit(df)
    eigen_values, _ = gfa.get_eigenvalues()
    plot_eisen_values(df, eigen_values)


def EFA_analysis(): 
    ''' Purpose: Complete Exploratory Factor Analysis. '''
    df = load_CSV("data", ["subno"])
    headers = list(df.columns.values)
    efa = FactorAnalyzer()
    efa.set_params(n_factors=2, rotation="varimax")
    efa.fit(df)
    plot_efa_results(efa, headers)
    summary = efa.get_factor_variance()
    print(f"Correlation Scores: {efa.loadings_}")
    print(f"Variance:{summary[0]}")
    print(f"Proportional Variance:{summary[1]}")
    print(f"Cumulative Variance:{summary[2]}")


determine_topics()
EFA_analysis()


