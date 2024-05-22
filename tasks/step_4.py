from code import Monitor, ComparedModel, load_test_pair, compare
from matplotlib import pyplot
from numpy import array, arange, linspace, where
from os import path, mkdir
from PIL import Image
from skimage import io
from tasks import save_data, load_data
from torch import cuda

monitor, device = Monitor(), "cuda:0" if cuda.is_available() else "cpu"


def task():
    if not path.exists("../outputs/task4/"):
        mkdir("../outputs/task4/")

    # sam_vit_h_4b8939.pth, sam_vit_l_0b3195.pth, and sam_vit_b_01ec64.pth:
    # These are models with different variants of the Vision Transformer (ViT) backbone.
    # The 'h', 'l', and 'b' denote the size of the ViT backbone ('h' for huge, 'l' for large, and 'b' for base).
    # Typically, the 'h' model would give you the best results but at the cost of running time and memory usage
    # because of the large transformer architecture. Conversely, the 'b' model would be fastest and
    # least memory-intensive but might not provide the best results.
    mask_data, info_data = load_test_pair(verbose=True)
    scores = {"base": [], "large": [], "huge": [], "medsam": [], "cellpose": []}

    print("run the test for baseline SAM.")
    for process_index, (expected_mask, image) in enumerate(zip(mask_data, info_data)):
        save_path = "../outputs/task4/b." + str(process_index + 1).zfill(len(str(len(info_data)))) + ".pkl"
        if not path.exists(save_path):
            model = ComparedModel(model_type="vit_b")
            stats = compare(expected_mask=expected_mask, obtained_mask=model(image))
            save_data(save_path=save_path, information=stats)
        else:
            stats = load_data(load_path=save_path)
        scores["base"].append(stats["f1"])
        monitor(process_index + 1, len(info_data), extra={"F1 score": "%.4f" % stats["f1"]})

    print("run the test for low resolution SAM.")
    for process_index, (expected_mask, image) in enumerate(zip(mask_data, info_data)):
        save_path = "../outputs/task4/l." + str(process_index + 1).zfill(len(str(len(info_data)))) + ".pkl"
        if not path.exists(save_path):
            model = ComparedModel(model_type="vit_l")
            stats = compare(expected_mask=expected_mask, obtained_mask=model(image))
            save_data(save_path=save_path, information=stats)
        else:
            stats = load_data(load_path=save_path)

        if process_index + 1 in [22, 23]:
            save_path = "../outputs/task4/" + str(process_index + 1).zfill(len(str(len(info_data)))) + ".tiff"
            if not path.exists(save_path):
                model = ComparedModel(model_type="vit_l")
                mask = model(image)
                io.imsave(save_path, mask, check_contrast=False)
        scores["large"].append(stats["f1"])
        monitor(process_index + 1, len(info_data), extra={"F1 score": "%.4f" % stats["f1"]})

    print("run the test for high resolution SAM.")
    for process_index, (expected_mask, image) in enumerate(zip(mask_data, info_data)):
        save_path = "../outputs/task4/h." + str(process_index + 1).zfill(len(str(len(info_data)))) + ".pkl"
        if not path.exists(save_path):
            model = ComparedModel(model_type="vit_h")
            stats = compare(expected_mask=expected_mask, obtained_mask=model(image))
            save_data(save_path=save_path, information=stats)
        else:
            stats = load_data(load_path=save_path)
        scores["huge"].append(stats["f1"])
        monitor(process_index + 1, len(info_data), extra={"F1 score": "%.4f" % stats["f1"]})

    for process_index in range(len(mask_data)):
        save_path = "../outputs/task1/" + str(process_index + 1).zfill(len(str(len(info_data)))) + ".pkl"
        stats = load_data(load_path=save_path)
        scores["cellpose"].append(stats["f1"])

    for process_index in range(len(mask_data)):
        save_path = "../outputs/task2/" + str(process_index + 1).zfill(len(str(len(info_data)))) + ".pkl"
        stats = load_data(load_path=save_path)
        scores["medsam"].append(stats["f1"])

    return scores


def show(scores):
    for key in scores.keys():
        scores[key] = array(scores[key])

    figure = pyplot.figure(figsize=(10, 10), tight_layout=True)
    ax = pyplot.subplot(3, 2, 1)
    values = array([
        len(where(scores["base"] == 0)[0]), len(where(scores["large"] == 0)[0]), len(where(scores["huge"] == 0)[0]),
        len(where(scores["medsam"] == 0)[0]), len(where(scores["cellpose"] == 0)[0])
    ])
    pyplot.bar(arange(len(values)), values / 450.0 + 0.06, bottom=-0.06, ec="k", fc="silver", lw=0.75)
    for index, value in enumerate(values):
        pyplot.text(index, value / 450.0 + 0.01, ("%.1f" % (value / 450.0 * 100.0)) + "%",
                    va="bottom", ha="center")
    pyplot.xlabel("model")
    pyplot.ylabel("no mask response")
    pyplot.xticks(arange(len(values)), ["SAM-B", "SAM-L", "SAM-H", "MedSAM", "Cellpose"])
    pyplot.yticks(linspace(0, 1, 11), [str(v) + "%" for v in arange(0, 101, 10)])
    pyplot.xlim(-0.6, len(values) - 0.4)
    pyplot.ylim(-0.06, 1.06)
    # noinspection PyUnresolvedReferences
    ax.spines["top"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["right"].set_visible(False)

    ax = pyplot.subplot(3, 2, 2)
    for index, label in enumerate(["base", "large", "huge", "cellpose"]):
        values = scores[label]
        pyplot.boxplot([values[values > 0]], positions=[index], notch=True)
    pyplot.xlabel("model")
    pyplot.ylabel("F1 score")
    pyplot.xticks(arange(4), ["SAM-B", "SAM-L", "SAM-H", "Cellpose"])
    pyplot.yticks(linspace(0, 1, 11), [str(v) + "%" for v in arange(0, 101, 10)])
    pyplot.xlim(-0.6, 3.6)
    pyplot.ylim(-0.06, 1.06)
    # noinspection PyUnresolvedReferences
    ax.spines["top"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["right"].set_visible(False)

    pyplot.subplot(3, 2, 3)
    pyplot.fill_between([0, 1], [0, 0], [0, 1], fc="#EEEEEE", lw=0, zorder=0)
    pyplot.plot([0, 1], [0, 1], color="k", lw=0.75, ls="--", zorder=2)
    pyplot.plot([0.0, 0.5, 0.5], [0.5, 0.5, 0.0], color="k", lw=0.75, ls="--", zorder=2)
    rate, count = 0, 0
    for v_1, v_2 in zip(scores["large"], scores["base"]):
        if v_1 > 0 and v_2 > 0:
            count += 1
            if v_1 > v_2:
                pyplot.scatter([v_1], [v_2], color="red", alpha=0.5, zorder=0)
                rate += 1
            elif v_2 > v_1:
                pyplot.scatter([v_1], [v_2], color="blue", alpha=0.5, zorder=0)
            else:
                pyplot.scatter([v_1], [v_2], color="gray", alpha=0.5, zorder=0)

    pyplot.text(1.00, 0.00, ("%.1f" % (rate / count * 100)) + "%", va="bottom", ha="right")
    pyplot.xlabel("F1 score from SAM-B")
    pyplot.ylabel("F1 score from SAM-L")
    pyplot.xticks(linspace(0, 1, 11), ["%.1f" % v for v in linspace(0, 1, 11)])
    pyplot.yticks(linspace(0, 1, 11), ["%.1f" % v for v in linspace(0, 1, 11)])
    pyplot.xlim(-0.033, 1.033)
    pyplot.ylim(-0.050, 1.050)

    pyplot.subplot(3, 2, 4)
    pyplot.fill_between([0, 1], [0, 0], [0, 1], fc="#EEEEEE", lw=0, zorder=0)
    pyplot.plot([0, 1], [0, 1], color="k", lw=0.75, ls="--", zorder=2)
    pyplot.plot([0.0, 0.5, 0.5], [0.5, 0.5, 0.0], color="k", lw=0.75, ls="--", zorder=2)
    rate, count = 0, 0
    for v_1, v_2 in zip(scores["huge"], scores["large"]):
        if v_1 > 0 and v_2 > 0:
            count += 1
            if v_1 > v_2:
                pyplot.scatter([v_1], [v_2], color="red", alpha=0.5, zorder=0)
                rate += 1
            elif v_2 > v_1:
                pyplot.scatter([v_1], [v_2], color="blue", alpha=0.5, zorder=0)
            else:
                pyplot.scatter([v_1], [v_2], color="gray", alpha=0.5, zorder=0)
    pyplot.text(1.00, 0.00, ("%.1f" % (rate / count * 100)) + "%", va="bottom", ha="right")
    pyplot.xlabel("F1 score from SAM-L")
    pyplot.ylabel("F1 score from SAM-H")
    pyplot.xticks(linspace(0, 1, 11), ["%.1f" % v for v in linspace(0, 1, 11)])
    pyplot.yticks(linspace(0, 1, 11), ["%.1f" % v for v in linspace(0, 1, 11)])
    pyplot.xlim(-0.033, 1.033)
    pyplot.ylim(-0.050, 1.050)

    ax = pyplot.subplot(3, 2, 5)
    ax.imshow(Image.open("../datasets/test/059.h.image.tiff"), cmap="gray", extent=(-0.45, -0.05, 0.6, 1.0))
    ax.imshow(Image.open("../datasets/test/077.h.image.tiff"), cmap="gray", extent=(+0.05, +0.45, 0.6, 1.0))
    ax.imshow(Image.open("../datasets/test/150.h.image.tiff"), cmap="gray", extent=(-0.45, -0.05, 0.1, 0.5))
    ax.imshow(Image.open("../datasets/test/221.h.image.tiff"), cmap="gray", extent=(+0.05, +0.45, 0.1, 0.5))
    ax.imshow(Image.open("../datasets/test/001.o.image.tiff"), cmap="gray", extent=(+0.55, +0.95, 0.6, 1.0))
    ax.imshow(Image.open("../datasets/test/004.o.image.tiff"), cmap="gray", extent=(+1.05, +1.45, 0.6, 1.0))
    ax.imshow(Image.open("../datasets/test/074.h.image.tiff"), cmap="gray", extent=(+0.55, +0.95, 0.1, 0.5))
    ax.imshow(Image.open("../datasets/test/103.h.image.tiff"), cmap="gray", extent=(+1.05, +1.45, 0.1, 0.5))
    pyplot.vlines(0.5, 0.0, 1.0, color="k", lw=0.75, ls="--", zorder=0)
    pyplot.xlabel("case analysis for images")
    pyplot.xticks([0, 1], ["detectable (>0.5)", "undetectable (<0.1)"])
    pyplot.yticks([])
    pyplot.xlim(-0.5, 1.5)
    pyplot.ylim(0.0, 1.0)

    # noinspection PyUnresolvedReferences
    ax.spines["top"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["right"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["left"].set_visible(False)

    ax = pyplot.subplot(3, 2, 6)
    ax.imshow(Image.open("../datasets/test/022.o.image.tiff"), extent=(-0.30, +0.30, 0.6, 1.0))
    ax.imshow(Image.open("../datasets/test/023.o.image.tiff"), extent=(-0.30, +0.30, 0.1, 0.5))
    ax.imshow(Image.open("../datasets/test/022.o.label.tiff"), cmap="gray", extent=(+0.70, +1.30, 0.6, 1.0))
    ax.imshow(Image.open("../datasets/test/023.o.label.tiff"), cmap="gray", extent=(+0.70, +1.30, 0.1, 0.5))
    ax.imshow(Image.open("../outputs/task4/022.tiff"), cmap="gray", extent=(+1.70, +2.30, 0.6, 1.0))
    ax.imshow(Image.open("../outputs/task4/023.tiff"), cmap="gray", extent=(+1.70, +2.30, 0.1, 0.5))
    pyplot.xlabel("case analysis for image-mask pairs")
    pyplot.xticks([0, 1, 2], ["image", "expected mask", "obtained mask"])
    pyplot.yticks([])
    pyplot.xlim(-0.5, 2.5)
    pyplot.ylim(0, 1)

    # noinspection PyUnresolvedReferences
    ax.spines["top"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["right"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["left"].set_visible(False)

    figure.align_labels()
    figure.text(0.02, 0.99, "a", va="center", ha="center", fontsize=12)
    figure.text(0.52, 0.99, "b", va="center", ha="center", fontsize=12)
    figure.text(0.02, 0.66, "c", va="center", ha="center", fontsize=12)
    figure.text(0.52, 0.66, "d", va="center", ha="center", fontsize=12)
    figure.text(0.02, 0.33, "e", va="center", ha="center", fontsize=12)
    figure.text(0.52, 0.33, "f", va="center", ha="center", fontsize=12)

    pyplot.savefig("../outputs/result-03.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


if __name__ == "__main__":
    show(task())
