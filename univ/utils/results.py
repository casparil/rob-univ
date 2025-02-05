import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt


def load_csvs(path: str, num_experiments: int = 10, result_type: str = ''):
    """
    Loads the computed similarity results saved as CSV files from the given folder. They file names are expected to
    correspond to the given path file name with an additional experiment number at the end. If multiple experiments
    have been conducted, their results are concatenated in a single pandas DataFrame.

    :param path: The path to the file.
    :param num_experiments: The number of experiments conducted.
    :param result_type: The type of file to be loaded, i.e. regular scores, baseline scores etc.
    :return: The DataFrame containing the concatenated results.
    """
    df = None
    for num in range(num_experiments):
        if df is None:
            df = pd.read_csv(path.format(result_type, num), index_col=0)
        else:
            df = pd.concat((df, pd.read_csv(path.format(result_type, num), index_col=0)))
    return df


def plot_boxplots(df: pd.DataFrame, models: list, rows: int = 4, cols: int = 2, figsize: tuple = (20, 20),
                  title: str = 'Model Similarity'):
    """
    Plots boxplots for the given DataFrame containing concatenated similarity results for different models. For each
    model in the model list, its own boxplots containing all similarity results computed between the model itself and
    all other models are displayed.

    :param df: The DataFrame containing the data.
    :param models: A list of model names.
    :param rows: The number of rows the figure should have, default: 4.
    :param cols: The number of columns the figure should have, default: 2.
    :param figsize: The size of the figure, default: (20, 20).
    :param title: The title of the figure, default: 'Model Similarity'.
    """
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    for idx, name in enumerate(models):
        df_plot = df[[name, 'models']]
        sns.boxplot(x='models', y=name, data=df_plot[df_plot.models != name], ax=ax[idx // cols][idx % cols])
    fig.suptitle(title, y=0.9)
    plt.show()


def remove_self_sim_scores(df: pd.DataFrame, num_models: int = 8):
    """
    Removes the self-similarity scores of models from the given dataframe. The scores are removed for each row based on
    the current row number and the number of models for which results were calculated.

    :param df: The dataframe containing the similarity scores.
    :param num_models: The number of models for which results were calculated, default: 8.
    """
    array = df.to_numpy()
    res = []
    for result in range(len(array)):
        res.append(np.delete(array[result], result % num_models))
    return np.array(res)
