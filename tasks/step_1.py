from code import Monitor, BaselineModel, load_test_pair, compare
from matplotlib import pyplot
from numpy import arange, linspace, mean
from os import path, mkdir
from tasks import save_data, load_data


def task():
    reported_scores = [0.0000, 0.0000, 0.0000, 0.0000, 0.0218, 0.0851, 0.8889, 0.0769, 0.4928, 0.5849, 0.4706, 0.3738,
                       0.5688, 0.5676, 0.5946, 0.9923, 0.8627, 0.9103, 0.8806, 0.9259, 0.9449, 0.9309, 0.9062, 0.9266,
                       0.9146, 0.9402, 0.9442, 0.9487, 0.9532, 0.9528, 0.9391, 0.9437, 0.9163, 0.9438, 0.9438, 0.9434,
                       0.9470, 0.9189, 0.8976, 0.9173, 0.9394, 0.9385, 0.9457, 0.9206, 0.9134, 0.9291, 0.9291, 0.9304,
                       0.9562, 0.9778, 0.9738, 0.9498, 0.9736, 0.9658, 0.9579, 0.9688, 0.9741, 0.9741, 0.9579, 0.9255,
                       0.8550, 0.8696, 0.9147, 0.9277, 0.9917, 0.9793, 0.9950, 0.9336, 0.9012, 0.9322, 0.9100, 0.9180,
                       0.9211, 0.9570, 0.9419, 0.1784, 0.2243, 0.3196, 0.3051, 0.6522, 0.0000, 0.0000, 0.0000, 0.8632,
                       0.1053, 0.0000, 0.0000, 0.5000, 0.7451, 0.0000, 0.2857, 0.3170, 0.2646, 0.3944, 0.8627, 0.4545,
                       0.0000, 0.1333, 0.8621, 0.0000, 0.0000, 0.4167, 0.0000, 0.7895, 0.1811, 0.2302, 0.2778, 0.2705,
                       0.1333, 0.1250, 0.0000, 0.0000, 0.7959, 0.8387, 0.8364, 0.0000, 0.0000, 0.4865, 0.8050, 0.0000,
                       0.0000, 0.1924, 0.2714, 0.2149, 0.1648, 0.6190, 0.5000, 0.4167, 0.0000, 0.6957, 0.8276, 0.9444,
                       0.8571, 0.0000, 0.5294, 0.3876, 0.2383, 0.2637, 0.3455, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                       0.0000, 0.0000, 0.2315, 0.5845, 0.4439, 0.1940, 0.2659, 0.3278, 0.2627, 0.4233, 0.5115, 0.4333,
                       0.6186]

    if not path.exists("../outputs/task1/"):
        mkdir("../outputs/task1/")

    mask_data, info_data = load_test_pair(folder_path="../dataset/test/", verbose=True)

    print("run the test.")
    model, monitor, investigated_scores = BaselineModel(), Monitor(), []
    for process_index, (expected_mask, image) in enumerate(zip(mask_data, info_data)):
        save_path = "../outputs/task1/" + str(process_index + 1).zfill(len(str(len(info_data)))) + ".pkl"
        if not path.exists(save_path):
            obtained_mask = model(image=image)
            stats = compare(expected_mask=expected_mask, obtained_mask=obtained_mask)
            save_data(save_path=save_path, information=stats)
        else:
            stats = load_data(load_path=save_path)

        investigated_scores.append(stats["f1"])
        monitor(process_index + 1, len(info_data))

    figure = pyplot.figure(figsize=(10, 5), tight_layout=True)

    ax = pyplot.subplot(1, 2, 1)
    violin = pyplot.violinplot([reported_scores, investigated_scores],
                               positions=[1, 2], widths=0.5, showextrema=False)
    # noinspection PyTypeChecker
    for patch in violin["bodies"]:
        patch.set_edgecolor("black")
        patch.set_facecolor("silver")
        patch.set_linewidth(0.75)
        patch.set_alpha(1.00)
    pyplot.scatter([1, 2], [mean(reported_scores), mean(investigated_scores)],
                   s=30, fc="w", ec="k", lw=0.75, zorder=3)
    pyplot.hlines(mean(reported_scores), 0.6, 1.4, lw=1.5, color="k", zorder=2)
    pyplot.hlines(mean(investigated_scores), 1.6, 2.4, lw=1.5, color="k", zorder=2)
    pyplot.text(1, 1.01, "%d samples" % len(reported_scores), ha="center", va="bottom")
    pyplot.text(2, 1.01, "%d samples" % len(investigated_scores), ha="center", va="bottom")
    pyplot.text(1, mean(reported_scores) + 0.01, "%.2f" % mean(reported_scores), ha="center", va="bottom")
    pyplot.text(2, mean(investigated_scores) + 0.01, "%.2f" % mean(investigated_scores), ha="center", va="bottom")
    pyplot.xlabel("performance of cellpose")
    pyplot.ylabel("F1 score")
    pyplot.xticks([1, 2], ["reported", "investigated"])
    pyplot.yticks(linspace(0, 1, 11), ["%.1f" % v for v in linspace(0, 1, 11)])
    pyplot.xlim(0.4, 2.6)
    pyplot.ylim(-0.05, 1.05)
    # noinspection PyUnresolvedReferences
    ax.spines["top"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["right"].set_visible(False)

    ax = pyplot.subplot(1, 2, 2)
    values_1, values_2 = sorted(reported_scores), sorted(investigated_scores)
    collections, last, zero_count = [], 0, 0
    for former_index, former_value in enumerate(values_1):
        if former_value > 0:
            for latter_index, latter_value in zip(arange(len(values_2))[last:], values_2[last:]):
                if latter_value >= former_value:
                    if latter_value - former_value < 0.01:
                        last = latter_index
                        collections.append([[former_index, latter_index], [former_value, latter_value]])
                    break
        else:
            zero_count += 1
    for collection, color in zip(collections, pyplot.get_cmap("rainbow")(linspace(0, 1, len(collections)))):
        pyplot.plot(collection[0], collection[1], color=color, lw=0.75, zorder=0)
    pyplot.plot(arange(len(values_1)), values_1, color="black", label="reported", zorder=3)
    pyplot.plot(arange(len(values_2)), values_2, color="gray", label="investigated", zorder=2)

    pyplot.text(33, 0.75, "(" + str(zero_count + len(collections)) + "/" + str(len(reported_scores)) + ")\naligns",
                ha="center", va="center")
    pyplot.legend(loc="lower right")
    pyplot.xlabel("sample index (ordered by F1 score)")
    pyplot.ylabel("F1 score")
    pyplot.xticks(arange(0, 451, 50), arange(0, 451, 50))
    pyplot.yticks(linspace(0, 1, 11), ["%.1f" % v for v in linspace(0, 1, 11)])
    pyplot.xlim(-20, 470)
    pyplot.ylim(-0.05, 1.05)
    # noinspection PyUnresolvedReferences
    ax.spines["top"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["right"].set_visible(False)

    figure.align_labels()
    figure.text(0.02, 0.99, "a", va="center", ha="center", fontsize=12)
    figure.text(0.51, 0.99, "b", va="center", ha="center", fontsize=12)

    pyplot.savefig("../outputs/result-01.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


if __name__ == "__main__":
    task()
