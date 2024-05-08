from code import Monitor, ComparedModel, load_test_pair, compare
from matplotlib import pyplot
from numpy import array, linspace, arange, repeat
from os import path, mkdir
from skimage import io
from tasks import save_data, load_data


def task():
    if not path.exists("../outputs/task2/"):
        mkdir("../outputs/task2/")

    mask_data, info_data = load_test_pair(folder_path="../dataset/test/", verbose=True)

    print("run the test.")
    model, monitor, f1_scores, case = ComparedModel(), Monitor(), [], None
    for process_index, (expected_mask, image) in enumerate(zip(mask_data, info_data)):
        save_path = "../outputs/task2/" + str(process_index + 1).zfill(len(str(len(info_data)))) + ".pkl"
        if not path.exists(save_path):
            obtained_mask = model(image=image)
            stats = compare(expected_mask=expected_mask, obtained_mask=obtained_mask)
            save_data(save_path=save_path, information=stats)
        else:
            stats = load_data(load_path=save_path)

        f1_scores.append(stats["f1"])
        monitor(process_index + 1, len(info_data))

    figure = pyplot.figure(figsize=(10, 5), tight_layout=True)
    grid = pyplot.GridSpec(2, 4)

    x, y = array([95, 190, 190, 95, 95]), array([255, 255, 350, 350, 255])
    # noinspection PyUnresolvedReferences
    pyplot.subplot(grid[0, 0])
    image = repeat(io.imread("../dataset/demo/medsam.png")[:, :, None], 3, axis=-1)
    pyplot.plot(x, y, lw=0.75, ls="--", c="k")
    pyplot.imshow(image)
    pyplot.axis("off")

    # noinspection PyUnresolvedReferences
    pyplot.subplot(grid[0, 1])
    image = (1 - repeat(io.imread("../dataset/demo/seg_medsam.png")[:, :, None], 3, axis=-1)) * 255
    pyplot.plot(x, y, lw=0.75, ls="--", c="k")
    pyplot.imshow(image)
    pyplot.xticks([])
    pyplot.yticks([])

    x, y = array([440, 640, 640, 440, 440]), array([440, 440, 640, 640, 440])

    # noinspection PyUnresolvedReferences
    pyplot.subplot(grid[1, 0])
    image = io.imread("../dataset/demo/challenge.tiff")
    pyplot.imshow(image)
    pyplot.plot(x, y, lw=0.75, ls="--", c="k")
    pyplot.axis("off")

    # noinspection PyUnresolvedReferences
    pyplot.subplot(grid[1, 1])
    image = (1 - repeat(io.imread("../dataset/demo/seg_challenge.tiff")[:, :, None], 3, axis=-1)) * 255
    pyplot.imshow(image)
    pyplot.plot(x, y, lw=0.75, ls="--", c="k")
    pyplot.xticks([])
    pyplot.yticks([])

    # noinspection PyUnresolvedReferences
    ax = pyplot.subplot(grid[:, 2:])
    pyplot.text(450 / 2, 0.01, "all results are 0", ha="center", va="bottom")
    pyplot.plot(arange(len(f1_scores)), sorted(f1_scores), color="black", label="reported", zorder=3)
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
    figure.text(0.28, 0.99, "b", va="center", ha="center", fontsize=12)
    figure.text(0.02, 0.54, "c", va="center", ha="center", fontsize=12)
    figure.text(0.28, 0.54, "d", va="center", ha="center", fontsize=12)
    figure.text(0.49, 0.99, "e", va="center", ha="center", fontsize=12)

    pyplot.savefig("../outputs/result-02.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


if __name__ == "__main__":
    task()