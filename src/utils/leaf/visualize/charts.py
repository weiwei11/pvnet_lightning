# Author: weiwei
import numpy as np
from matplotlib import pyplot as plt

# plt.style.use('ggplot')


def set_title_and_label(title=None, x_label=None, y_label=None):
    if title is not None:
        plt.title(title)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)


def plot_auc(x, ys, g_labels=None, x_label=None, y_label=None, title=None):
    handles = plt.plot(x, ys)
    if g_labels is not None:
        plt.legend(handles=handles, labels=g_labels)
    set_title_and_label(title, x_label, y_label)
    return handles


if __name__ == '__main__':
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.4, 0.6, 0.7, 0.8, 0.9, 1.1])
    y2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1])
    y = np.column_stack([y, y2])
    # plot_auc(x, y, ['x', 'y'])
    # plt.show()

    def gaussian_kernel(distance, bandwidth):
        return (1 / (bandwidth * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (distance / bandwidth) ** 2)
    a = np.arange(-180, 0)
    x = 1 - np.cos(a / 180 * np.pi)
    # x = np.arange(-100, 100)
    y = gaussian_kernel(x, 0.05)
    print(y)
    plt.plot(x, y)
    plt.show()
