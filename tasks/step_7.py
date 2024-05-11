from matplotlib import pyplot
from numpy import array, linspace, arange, mean
from os import listdir
from tasks import load_data


def show(batch_number: int):
    record_1, size = {}, len(listdir("../datasets/test/")) // 2
    for flag in ["b", "l", "h"]:
        results = []
        for sample_index in range(1, size + 1):
            load_path = "../outputs/task4/" + flag + "." + str(sample_index).zfill(len(str(size))) + ".pkl"
            results.append(load_data(load_path=load_path)["f1"])
        record_1[flag] = array(results)

    record_2 = []
    for batch_index in range(1, batch_number + 1):
        batch_info, results = str(batch_index).zfill(len(str(batch_number))), []
        for sample_index in range(1, size + 1):
            sample_info = str(sample_index).zfill(len(str(size)))
            load_path = "../outputs/task6/" + batch_info + "." + sample_info + ".pkl"
            results.append(load_data(load_path=load_path)["f1"])
        record_2.append(array(results))

    figure = pyplot.figure(figsize=(10, 5), tight_layout=True)
    ax = pyplot.subplot(1, 2, 1)
    violin = pyplot.violinplot([record_1["b"], record_1["l"], record_1["h"]],
                               positions=[1, 2], showextrema=False)
    # noinspection PyTypeChecker
    for patch in violin["bodies"]:
        patch.set_edgecolor("black")
        patch.set_facecolor("silver")
        patch.set_linewidth(0.75)
        patch.set_alpha(1.00)
    pyplot.xlabel("type of segment-anything model")
    pyplot.ylabel("F1 score")
    pyplot.xticks([1, 2, 3], ["baseline", "low-resolution", "high-resolution"])
    pyplot.yticks(linspace(0, 1, 11), ["%.1f" % v for v in linspace(0, 1, 11)])
    pyplot.xlim(0.4, 3.6)
    pyplot.ylim(-0.05, 1.05)
    # noinspection PyUnresolvedReferences
    ax.spines["top"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["right"].set_visible(False)

    ax = pyplot.subplot(1, 2, 2)
    for index in range(batch_number):
        values = record_2[index]
        if mean(values) > 0.55:
            pyplot.boxplot([values], positions=[index + 1], vert=True, notch=True, patch_artist=True,
                           boxprops=dict(ec="b", fc="#FCB1AB", lw=0.5))
        else:
            pyplot.boxplot([values], positions=[index + 1], vert=True, notch=True, patch_artist=True,
                           boxprops=dict(ec="b", fc="#FCE0AB", lw=0.5))

    pyplot.xlabel("type of segment-anything model")
    pyplot.ylabel("F1 score")
    pyplot.xticks(arange(0, 41, 5), arange(0, 41, 5))
    pyplot.yticks(linspace(0, 1, 11), ["%.1f" % v for v in linspace(0, 1, 11)])
    pyplot.xlim(0, batch_number + 1)
    pyplot.ylim(-0.05, 1.05)
    # noinspection PyUnresolvedReferences
    ax.spines["top"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["right"].set_visible(False)

    figure.align_labels()
    figure.text(0.02, 0.99, "a", va="center", ha="center", fontsize=12)
    figure.text(0.53, 0.99, "b", va="center", ha="center", fontsize=12)

    pyplot.savefig("../outputs/result-03.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


if __name__ == "__main__":
    show(batch_number=40)
