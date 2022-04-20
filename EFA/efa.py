# Creation Date: 18/04/2022


import numpy
import pandas
import itertools
import matplotlib
import matplotlib.pyplot
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo


# Matplot setup
matplotlib.rcParams['figure.figsize'] = (10.0, 6.0)
# matplotlib.use('Agg')


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


def load_CSV(filename: str, drop_list: list):
    ''' Returns: CSV loaded to dataframe with select columns dropped. '''
    dataframe = pandas.read_csv(f"{filename}.csv")
    if drop_list is not None:
        dataframe.drop(drop_list, axis=1, inplace=True)
    return dataframe


def adequacy_test(data):
    ''' Purpose: Measures adequacy of data for factorability. '''
    chi_square_score, p_value = calculate_bartlett_sphericity(
        data)  # Bartlett's Test
    _, kmo_score = calculate_kmo(data)  # Kaiser-Meyer-Olkin Test
    print(f"Bartlett's Test: {(chi_square_score, p_value)}")
    print(f"Kaiser-Meyer-Olkin Test: {(kmo_score)}")


def plot_eisen_values(filename: str, eigen_values: list, width: int):
    ''' Purpose: Creates visualisation of eisenvalue results. '''
    matplotlib.pyplot.figure(figsize=(8, 6))
    matplotlib.pyplot.axhline(y=1, c='k', linestyle="dashdot", alpha=0.4)
    matplotlib.pyplot.scatter(range(1, width+1), eigen_values, c="b")
    matplotlib.pyplot.plot(
        range(1, width+1), eigen_values, "b", label="PAF - Data")
    matplotlib.pyplot.title('Eigenvalue Analysis', {'fontsize': 20})
    matplotlib.pyplot.xlabel('Factor', {'fontsize': 15})
    matplotlib.pyplot.ylabel('Eigenvalue', {'fontsize': 15})
    matplotlib.pyplot.legend()
    matplotlib.pyplot.savefig(f"EFA/{filename}_eisenvalues.png")


def plot_horn_results(filename: str, eigenvalues: list, average_paf_eigens: list,
                      width: int):
    ''' Purpose: Creates visualisation of parallel analysis results. '''
    matplotlib.pyplot.figure(figsize=(8, 6))
    matplotlib.pyplot.axhline(y=1, c='k', linestyle="dashdot", alpha=0.4)
    matplotlib.pyplot.plot(
        range(1, width+1), average_paf_eigens, 'b', label='PAF - random', alpha=0.4)
    matplotlib.pyplot.scatter(
        range(1, width+1), eigenvalues, c='b', marker='o')
    matplotlib.pyplot.plot(
        range(1, width+1), eigenvalues, 'b', label='PAF - data')
    matplotlib.pyplot.title('Parallel Analysis', {'fontsize': 20})
    matplotlib.pyplot.xlabel('Factor', {'fontsize': 15})
    matplotlib.pyplot.ylabel('Eigenvalue', {'fontsize': 15})
    matplotlib.pyplot.legend()
    matplotlib.pyplot.savefig(f"EFA/{filename}_parallel_analysis.png")


def horn_parallel_analysis(data, filename: str, times: int = 100):
    ''' Purpose: Performs horn parallel analysis to determine topics. '''
    height, width = data.shape
    paf = FactorAnalyzer(n_factors=1, method='principal', rotation=None)
    sum_paf_eigens = numpy.empty(width)
    for _ in range(0, times):
        paf.fit(numpy.random.normal(size=(height, width)))
        sum_paf_eigens = sum_paf_eigens + paf.get_eigenvalues()[0]
    average_paf_eigens = sum_paf_eigens / times
    paf.fit(data)
    eigenvalues = paf.get_eigenvalues()[0]
    print(f"Factor eigenvalues for random matrix:\n{average_paf_eigens}")
    print('Factor eigenvalues for data:\n', eigenvalues)
    suggestedFactors = sum((eigenvalues - average_paf_eigens) > 0)
    print(suggestedFactors)
    print("here")
    print(average_paf_eigens)
    plot_horn_results(filename, eigenvalues, average_paf_eigens, width)


def plot_scree(proportional_variance: list, filename: str, width: int):
    ''' Purpose: Creates visualisation of variance explained by factor. '''
    matplotlib.pyplot.figure(figsize=(8, 6))
    matplotlib.pyplot.plot(range(1, width+1), proportional_variance,
                           'o-', linewidth=2, color='blue', label="PAF - Data")
    matplotlib.pyplot.title('Variance Scree Plot', {'fontsize': 20})
    matplotlib.pyplot.xlabel('Factor', {'fontsize': 15})
    matplotlib.pyplot.ylabel('Variance Explained', {'fontsize': 15})
    matplotlib.pyplot.legend()
    matplotlib.pyplot.savefig(f"EFA/{filename}_variance_scree.png")


def determine_topics(data, filename: str, eisenvalue: bool, parallel: bool):
    ''' Purpose: Analysis results to determine topic number. '''
    adequacy_test(data)
    _, width = data.shape
    paf = FactorAnalyzer(n_factors=11, method='principal', rotation=None)
    paf.fit(data)
    if eisenvalue:
        eigen_values = paf.get_eigenvalues()[0]
        plot_eisen_values(filename, eigen_values, width)
    if parallel:
        horn_parallel_analysis(data, filename)
    summary = paf.get_factor_variance()
    print(f"Variance: {summary[0]}")
    print(f"Proportional Variance: {summary[1]}")
    print(f"Cumulative Variance: {summary[2]}")
    plot_scree(summary[1], filename, width)


def plot_paf_results(paf, filename: str, number_topics: int, headers: list):
    ''' Purpose: Creates visualisation of PAF results. '''
    GRID = numpy.abs(paf.loadings_)
    xlabels = [f"Factor {base26(x + 1)}" for x in range(number_topics)]
    figure, axis = matplotlib.pyplot.subplots()
    x = axis.pcolor(GRID, cmap="Blues")
    figure.colorbar(x, ax=axis)
    axis.set_yticks(numpy.arange(paf.loadings_.shape[0])+0.5, minor=False)
    axis.set_xticks(numpy.arange(paf.loadings_.shape[1])+0.5, minor=False)
    axis.set_yticklabels(headers)
    axis.set_xticklabels(xlabels)
    figure.tight_layout()
    matplotlib.pyplot.savefig(f"EFA/{filename}_factor_analysis.png")


# Rotation methods: https://factor-analyzer.readthedocs.io/en/latest/
def EFA_analysis(data, filename: str, number_topics: int, rotation_method: str):
    ''' Purpose: Complete Exploratory Factor Analysis. '''
    headers = list(data.columns.values)
    paf = FactorAnalyzer()
    paf.set_params(n_factors=number_topics, rotation=rotation_method)
    paf.fit(data)
    plot_paf_results(paf, filename, number_topics, headers)
    print(paf.get_factor_variance())
    print(f"Correlation Scores: \n{paf.loadings_}")


def EFA_pipeline(datafile: str, topic_list: list = [1], rotation_list: list = ["oblimin"],
                 eisenvalue: bool = True, parallel: bool = True, drop_list: list = None):
    ''' Purpose: Completes PAF based EFA given provided arguments. '''
    data = load_CSV(datafile, drop_list)
    determine_topics(data, f"EFA_{datafile}", eisenvalue, parallel)
    for topic_count, rotation in itertools.product(topic_list, rotation_list):
        EFA_analysis(data, f"EFA_{datafile}_{rotation}", topic_count, rotation)


EFA_pipeline("data", topic_list=[3], drop_list=["subno"])

topic_options = [2, 3]
rotation_options = ["oblimin", "promax"]
#EFA_pipeline(topic_options, rotation_options)
