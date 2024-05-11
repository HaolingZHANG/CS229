from code import load_tune_pair, batchize
from matplotlib import pyplot
from os import path, mkdir, listdir
from PIL import Image


def task(side: int,
         overlap: float,
         batch_capacity: int) \
        -> int:
    if not path.exists("../outputs/task3/"):
        mkdir("../outputs/task3/")

        mask_data, info_data = load_tune_pair(side=side, overlap=overlap, verbose=True)
        # classify images artificially for better investigation.
        classes = [[1, 3, 7, 8, 13, 14, 16, 18, 19, 20, 21, 22, 23, 24],
                   [2, 6, 10, 12, 25, 26],
                   [4, 5, 9, 11, 15, 17, 40, 41, 42, 70],
                   [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                    75, 76, 77, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
                    89, 90, 91, 92, 93, 94, 95, 96, 97, 98],
                   [43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                    56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
                    69, 101],
                   [71, 72, 73, 74, 78, 99, 100]]
        batch_number = batchize(labels=mask_data, images=info_data, classes=classes, side=side, overlap=overlap,
                                batch_capacity=batch_capacity, folder_path="../outputs/task3/", verbose=True)
    else:
        batch_number = len(list(listdir("../outputs/task3/")))

    return batch_number


def show(batch_number: int, batch_capacity: int):
    batch_index = str(1).zfill(len(str(batch_number)))
    pyplot.figure(figsize=(10, 10), tight_layout=True)
    for sample_index in range(1, batch_capacity + 1):
        pyplot.subplot(4, 4, sample_index)
        sample_index = str(sample_index).zfill(len(str(batch_capacity)))
        pyplot.title("image-" + str(sample_index).zfill(len(str(batch_capacity))))
        image_path = "../outputs/task3/" + batch_index + "/" + sample_index + ".image.tiff"
        pyplot.imshow(Image.open(image_path))
        pyplot.xticks([])
        pyplot.yticks([])
    pyplot.savefig("../outputs/data-1.png")
    pyplot.close()

    pyplot.figure(figsize=(10, 10), tight_layout=True)
    for sample_index in range(1, batch_capacity + 1):
        pyplot.subplot(4, 4, sample_index)
        sample_index = str(sample_index).zfill(len(str(batch_capacity)))
        pyplot.title("label-" + str(sample_index).zfill(len(str(batch_capacity))))
        image_path = "../outputs/task3/" + batch_index + "/" + sample_index + ".label.tiff"
        pyplot.imshow(Image.open(image_path))
        pyplot.xticks([])
        pyplot.yticks([])
    pyplot.savefig("../outputs/data-2.png")
    pyplot.close()


if __name__ == "__main__":
    show(batch_number=task(side=1024, overlap=0.1, batch_capacity=16), batch_capacity=16)
