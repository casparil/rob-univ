import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def plot_comparison(scores1: np.ndarray, scores2: np.ndarray, shape: tuple, names: list, title1: str = None,
                    title2: str = None, figsize: tuple = (20, 10)):
    """
    Plots two heatmaps for the given scores using the same colors. If titles are passed, the plot titles are set
    accordingly.

    :param scores1: The scores displayed in the first heatmap.
    :param scores2: The scores displayed in the second heatmap.
    :param shape: Tuple for reshaping the scores.
    :param names: A list of column and index names used for the heatmap, i.e. the model names.
    :param title1: The title of the first heatmap, default: None.
    :param title2: The title of the second heatmap, default: None.
    :param figsize: A tuple indicating the figure size, default: (20, 10).
    :return: The created dataframes
    """
    df1 = pd.DataFrame(scores1.reshape(shape), columns=names, index=names)
    df2 = pd.DataFrame(scores2.reshape(shape), columns=names, index=names)
    min1, min2 = np.min(scores1), np.min(scores2)
    max1, max2 = np.max(scores1), np.max(scores2)
    vmin = min1 if min1 < min2 else min2
    vmax = max1 if max1 > max2 else max2
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    sns.heatmap(data=df1, ax=ax[0], annot=True, vmin=vmin, vmax=vmax)
    sns.heatmap(data=df2, ax=ax[1], annot=True, vmin=vmin, vmax=vmax)
    if title1 is not None and title2 is not None:
        ax[0].set_title(title1)
        ax[1].set_title(title2)
    plt.show()
    return df1, df2


def plot_dataframes(scores1: pd.DataFrame, scores2: pd.DataFrame, scores3: pd.DataFrame, scores4: pd.DataFrame,
                    titles: list = None, figsize: tuple = (20, 23)):
    """
    Plots four heatmaps for the given scores using the same colors. If titles are passed, the plot titles are set
    accordingly.

    :param scores1: The scores displayed in the first heatmap.
    :param scores2: The scores displayed in the second heatmap.
    :param scores3: The scores displayed in the third heatmap.
    :param scores4: The scores displayed in the fourth heatmap.
    :param titles: A list of titles for the heatmaps, default: None.
    :param figsize: A tuple indicating the figure size, default: (20, 23).
    """
    assert titles is None or len(titles) == 4
    mins = [np.min(np.array(scores1)), np.min(np.array(scores2)), np.min(np.array(scores3)), np.min(np.array(scores4))]
    maxs = [np.max(np.array(scores1)), np.max(np.array(scores2)), np.max(np.array(scores3)), np.max(np.array(scores4))]
    vmin = np.min(mins)
    vmax = np.max(maxs)
    fig, ax = plt.subplots(2, 2, figsize=figsize)
    sns.heatmap(data=scores1, ax=ax[0][0], annot=True, vmin=vmin, vmax=vmax)
    sns.heatmap(data=scores2, ax=ax[0][1], annot=True, vmin=vmin, vmax=vmax)
    sns.heatmap(data=scores3, ax=ax[1][0], annot=True, vmin=vmin, vmax=vmax)
    sns.heatmap(data=scores4, ax=ax[1][1], annot=True, vmin=vmin, vmax=vmax)
    if titles:
        ax[0][0].set_title(titles[0])
        ax[0][1].set_title(titles[1])
        ax[1][0].set_title(titles[2])
        ax[1][1].set_title(titles[3])
    plt.show()


def plot_single_heatmap(scores: list, position: int = 0, figsize: tuple = (20, 20), cmap: str = 'rocket',
                        font_scale: float = 3, save_path: str = None):
    """
    Plots a heatmaps of one of the scores passed in the given list of dataframes extracting the minimum and maximum of
    all passed scores. The resulting heatmap can optionally be saved as a PDF file under the given path.

    :param scores: The list of scores containing the data to be plotted.
    :param position: The position of the scores which should be visualized.
    :param figsize: A tuple indicating the figure size, default: (20, 20).
    :param cmap: The colour scheme to use in the heatmap, default: 'rocket'.
    :param font_scale: The font size, default: 3.
    :param save_path: The path and filename under which the figure should be saved, default: None.
    """
    mins = [np.min(np.array(x)) for x in scores]
    maxs = [np.max(np.array(x)) for x in scores]
    vmin = np.min(mins)
    vmax = np.max(maxs)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(data=scores[position], ax=ax, annot=True, vmin=vmin, vmax=vmax, cmap=cmap)
    sns.set(font_scale=font_scale)
    plt.show()
    if save_path is not None:
        with PdfPages(save_path) as pdf:
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
