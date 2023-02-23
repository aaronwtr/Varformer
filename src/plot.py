import matplotlib.pyplot as plt
import numpy as np


def plot_bars(vector, title=None):
    x = np.arange(len(vector))
    vector = sorted(vector, reverse=True)
    plt.bar(x, vector)
    plt.xlabel('Gene')
    plt.ylabel('Tractability score')
    plt.title(title)
    plt.show()
