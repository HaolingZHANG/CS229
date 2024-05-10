from code import FineTunedModel, SegmentationDataset, load_tune_pair, batchize
from os import path, mkdir


def task(side: int, overlap: float, batch_capacity: int):
    if not path.exists("../outputs/task3/"):
        mkdir("../outputs/task3/")

    if not path.exists("../model/cell-sam-vit"):
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

        data_folder_path = "../outputs/task3/batches/"
        batch_number = batchize(labels=mask_data, images=info_data, classes=classes, side=side, overlap=overlap,
                                batch_capacity=batch_capacity, folder_path=data_folder_path, verbose=True)

        model = FineTunedModel()
        for batch_index in range(1, batch_number + 1):
            print("fine tune the model in batch %d." % batch_index)
            batch_path = data_folder_path + str(batch_index + 1).zfill(len(str(batch_number))) + "/"
            dataset = SegmentationDataset(batch_path=batch_path, batch_capacity=batch_capacity)


if __name__ == "__main__":
    task(side=1024, overlap=0.1, batch_capacity=64)
