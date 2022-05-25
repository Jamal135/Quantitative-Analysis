# Creation Date: 18/04/2022


import os
import numpy
import pandas
import matplotlib
import matplotlib.pyplot
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo


# Matplot setup
matplotlib.rcParams['figure.figsize'] = (10.0, 6.0)


def prepare_log(datafile: str, log: str):
    if log == None:
        reference = f"EFA/{datafile}.txt"
    else:
        reference = f"EFA/{log}.txt"
    try:
        os.remove(reference)
    except OSError:
        pass
    return open(reference, "a")


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


def adequacy_test(data, log):
    ''' Purpose: Measures adequacy of data for factorability. '''
    print(f"{'='*75}\nFactorability testing\n", file=log)
    chi_square_score, p_value = calculate_bartlett_sphericity(
        data)  # Bartlett's Test
    _, kmo_score = calculate_kmo(data)  # Kaiser-Meyer-Olkin Test
    print(f"Bartlett's Test: {(chi_square_score, p_value)}", file=log)
    print(f"Kaiser-Meyer-Olkin Test: ({kmo_score})", file=log)


def plot_eisen_values(filename: str, eigen_values: list, width: int):
    ''' Purpose: Creates visualisation of eisenvalue results. '''
    matplotlib.pyplot.figure(figsize=(8, 6))
    matplotlib.pyplot.axhline(y=1, c='k', linestyle="dashdot", alpha=0.4)
    matplotlib.pyplot.scatter(range(1, width+1), eigen_values, c="b")
    matplotlib.pyplot.plot(
        range(1, width+1), eigen_values, "b", label="PAF - Data")
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
    matplotlib.pyplot.xlabel('Factor', {'fontsize': 15})
    matplotlib.pyplot.ylabel('Eigenvalue', {'fontsize': 15})
    matplotlib.pyplot.legend()
    matplotlib.pyplot.savefig(f"EFA/{filename}_parallel_analysis.png")


def horn_parallel_analysis(data, filename: str, log: str, times: int = 100):
    ''' Purpose: Performs horn parallel analysis to determine topics. '''
    print(f"\n{'='*75}\nHorn parallel analysis\n", file=log)
    height, width = data.shape
    paf = FactorAnalyzer(n_factors=11, method='principal', rotation=None)
    sum_paf_eigens = numpy.empty(width)
    for _ in range(0, times):
        paf.fit(numpy.random.normal(size=(height, width)))
        sum_paf_eigens = sum_paf_eigens + paf.get_eigenvalues()[0]
    average_paf_eigens = sum_paf_eigens / times
    paf.fit(data)
    eigenvalues = paf.get_eigenvalues()[0]
    print(
        f"Factor eigenvalues for random matrix:\n{average_paf_eigens}\n", file=log)
    print(f"Factor eigenvalues for data:\n{eigenvalues}\n", file=log)
    suggested = sum((eigenvalues - average_paf_eigens) > 0)
    print(f"Parallel analysis suggested factors:{suggested}\n", file=log)
    plot_horn_results(filename, eigenvalues, average_paf_eigens, width)


def plot_scree(proportional_variance: list, filename: str, width: int):
    ''' Purpose: Creates visualisation of variance explained by factor. '''
    matplotlib.pyplot.figure(figsize=(8, 6))
    matplotlib.pyplot.plot(range(1, width+1), proportional_variance,
                           'o-', linewidth=2, color='blue', label="PAF - Data")
    matplotlib.pyplot.xlabel('Factor', {'fontsize': 15})
    matplotlib.pyplot.ylabel('Variance Explained', {'fontsize': 15})
    matplotlib.pyplot.legend()
    matplotlib.pyplot.savefig(f"EFA/{filename}_variance_scree.png")


def determine_topics(data, total_topics: int, filename: str, eisenvalue: bool,
                     parallel: bool, log: str):
    ''' Purpose: Analysis results to determine topic number. '''
    adequacy_test(data, log)
    _, width = data.shape
    paf = FactorAnalyzer(n_factors=total_topics,
                         method='principal', rotation=None)
    paf.fit(data)
    if eisenvalue:
        eigen_values = paf.get_eigenvalues()[0]
        print(f"\n{'='*75}\nEigenvalue analysis\n", file=log)
        print(f"Eigenvalues:\n{eigen_values}\n", file=log)
        suggested = next(x[0] for x in enumerate(eigen_values) if x[1] < 1)
        print(
            f"Eigenvalue suggested factors: {suggested}", file=log)
        plot_eisen_values(filename, eigen_values, width)
    if parallel:
        horn_parallel_analysis(data, filename, log)
    variance = paf.get_factor_variance()
    print(f"Variance\n{variance[0]}\n", file=log)
    print(f"Proportional variance\n{variance[1]}\n", file=log)
    print(f"Cumulative variance\n{variance[2]}", file=log)
    plot_scree(variance[1], filename, width)


def user_input():
    ''' Returns: User entered number of topics integer. '''
    while True:
        try:
            topic_count = int(input("Number of topics:"))
            break
        except:
            print("Enter a valid integer!")
    return topic_count


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
def EFA_analysis(data, filename: str, number_topics: int, rotation_method: str, log: str):
    ''' Purpose: Complete Exploratory Factor Analysis. '''
    headers = list(data.columns.values)
    paf = FactorAnalyzer()
    paf.set_params(n_factors=number_topics, rotation=rotation_method)
    paf.fit(data)
    plot_paf_results(paf, filename, number_topics, headers)
    print(f"Loading Scores:\n{paf.loadings_}\n", file=log)
    print(f"Communality Scores:\n{paf.get_communalities()}\n", file=log)
    print(f"Uniqueness Scores:\n{paf.get_uniquenesses()}\n", file=log)


def EFA_pipeline(datafile: str, rotation_list: list = ["oblimin"], eisenvalue: bool = True,
                 parallel: bool = True, drop_list: list = None, log: str = None):
    ''' Purpose: Completes PAF based EFA given provided arguments. '''
    LOG = prepare_log(datafile, log)
    data = load_CSV(datafile, drop_list)
    total_topics = len(list(data.columns.values))
    determine_topics(data, total_topics,
                     f"EFA_{datafile}", eisenvalue, parallel, LOG)
    topic_count = user_input()
    print(f"\n{'='*75}\nEFA\n\nTopic Number: {topic_count}\n", file=LOG)
    for rotation in rotation_list:
        print(f"EFA Rotation: {rotation}\n", file=LOG)
        EFA_analysis(
            data, f"EFA_{datafile}_{rotation}_{topic_count}", topic_count, rotation, LOG)


EFA_pipeline("data3", rotation_list=["oblimin", "promax"])
