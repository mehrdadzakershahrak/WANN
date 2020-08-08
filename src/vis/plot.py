import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_y_over_x(ys, xs, y_label, x_label,
                  line_labels, path, title,
                  line_colors, xticks=None):
    plt.clf()

    for i, x in enumerate(xs):
        plt.plot(xs[i], ys[i], label=line_labels[i],
                 color=line_colors[i], alpha=0.8)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()

    if xticks is not None:
        plt.xticks(np.arange(len(xticks)), xticks)

    plt.savefig(path)


def generate_heat_map(df, fields, target_field,
                      x_label, y_label, title,
                      path):
    plt.clf()
    plt.figure(figsize=(20, 20))

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    new_df = df.copy()
    new_df = np.round(new_df, decimals=5)
    new_df = new_df.pivot(fields[0], fields[1], target_field)
    sns.heatmap(new_df, annot=True, cmap="PuBuGn")

    plt.savefig(path)


def hist_plot(df, path):
    for c in df:
        plt.clf()
        df[c].hist()

        plt.savefig(path)
