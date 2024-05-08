from datetime import datetime
from matplotlib import pyplot
from numpy import ndarray, array, arange, min, max
from os import listdir
from PIL import Image, UnidentifiedImageError
from stardist.matching import matching
from typing import Tuple


class Monitor(object):

    def __init__(self):
        """
        Initialize the monitor to identify the task progress.
        """
        self.last_time = None

    def __call__(self,
                 moment: int,
                 finish: int,
                 extra: dict = None):
        """
        Output the current state of process.

        :param moment: current state of process.
        :type moment: int

        :param finish: total state of process.
        :type finish: int

        :param extra: extra vision information if required.
        :type extra: dict
        """
        if moment == 0 or self.last_time is None:
            self.last_time = datetime.now()
            return

        position = int(moment / finish * 100)

        string = "|"

        for index in range(0, 100, 5):
            if position >= index:
                string += "█"
            else:
                string += " "

        string += "|"

        pass_time = (datetime.now() - self.last_time).total_seconds()
        wait_time = int(pass_time * (finish - moment) / moment)

        string += " " * (3 - len(str(position))) + str(position) + "% ("

        string += " " * (len(str(finish)) - len(str(moment))) + str(moment) + "/" + str(finish)

        if moment < finish:
            minute, second = divmod(wait_time, 60)
            hour, minute = divmod(minute, 60)
            string += ") wait " + "%04d:%02d:%02d" % (hour, minute, second)
        else:
            minute, second = divmod(pass_time, 60)
            hour, minute = divmod(minute, 60)
            string += ") used " + "%04d:%02d:%02d" % (hour, minute, second)

        if extra is not None:
            string += " " + str(extra).replace("\"", "").replace("{", "(").replace("}", ")") + "."
        else:
            string += "."

        print("\r" + string, end="", flush=True)

        if moment >= finish:
            self.last_time = None
            print()


def load_train_pair(folder_path: str,
                    verbose: bool = False) \
        -> Tuple[list, list]:
    """
    Load train pairs from a folder.

    :param folder_path: test folder path.
    :type folder_path: str

    :param verbose: show the process.
    :type verbose: bool

    :return: test data pair.
    :rtype: list, list
    """
    value_1, value_2, indices, monitor = {}, {}, set(), Monitor()
    if verbose:
        print("load train images.")
        child_paths = list(listdir(folder_path))
        for process_index, child_path in enumerate(child_paths):
            try:
                image = array(Image.open(folder_path + child_path))
                if "label" in child_path:
                    index = (folder_path + child_path).split("_")[1]
                    value_1[index] = image
                else:
                    info = (folder_path + child_path).split("_")[1]
                    index = info[:info.rindex(".")]
                    value_2[index] = image
                indices.add(index)
            except UnidentifiedImageError:
                pass
            monitor(process_index + 1, len(child_paths))

        print("collect mask-image pairs.")
        mask_data, info_data, indices = [], [], sorted(list(indices))
        for process_index, ordered_index in enumerate(indices):
            if ordered_index in value_1 and ordered_index in value_2:
                mask_data.append(value_1[ordered_index])
                info_data.append(value_2[ordered_index])
            monitor(process_index + 1, len(indices))
    else:
        for child_path in listdir(folder_path):
            try:
                image = array(Image.open(folder_path + child_path))
                if "label" in child_path:
                    index = (folder_path + child_path).split("_")[1]
                    value_1[index] = image
                else:
                    info = (folder_path + child_path).split("_")[1]
                    index = info[:info.rindex(".")]
                    value_2[index] = image
                indices.add(index)
            except UnidentifiedImageError:
                pass

        mask_data, info_data = [], []
        for ordered_index in sorted(list(indices)):
            if ordered_index in value_1 and ordered_index in value_2:
                mask_data.append(value_1[ordered_index])
                info_data.append(value_2[ordered_index])

    return mask_data, info_data


def load_test_pair(folder_path: str,
                   verbose: bool = False) \
        -> Tuple[list, list]:
    """
    Load test pairs from a folder.

    :param folder_path: test folder path.
    :type folder_path: str

    :param verbose: show the process.
    :type verbose: bool

    :return: test data pair.
    :rtype: list, list
    """
    value_1, value_2, indices, monitor = {}, {}, set(), Monitor()
    if verbose:
        print("load test images.")
        chile_paths = list(listdir(folder_path))
        for process_index, child_path in enumerate(chile_paths):
            image, info = array(Image.open(folder_path + child_path)), child_path[:-13]
            if "label" in child_path:
                value_1[info] = image
            else:
                value_2[info] = image
            indices.add(info)
            monitor(process_index + 1, len(chile_paths))

        print("collect mask-image pairs.")
        mask_data, info_data, indices = [], [], sorted(list(indices))
        for process_index, ordered_index in enumerate(indices):
            if ordered_index in value_1 and ordered_index in value_2:
                mask_data.append(value_1[ordered_index])
                info_data.append(value_2[ordered_index])
            monitor(process_index + 1, len(indices))
    else:
        for child_path in listdir(folder_path):
            image, info = array(Image.open(folder_path + child_path)), child_path[:-13]
            if "label" in child_path:
                value_1[info] = image
            else:
                value_2[info] = image
            indices.add(info)

        mask_data, info_data = [], []
        for ordered_index in sorted(list(indices)):
            if ordered_index in value_1 and ordered_index in value_2:
                mask_data.append(value_1[ordered_index])
                info_data.append(value_2[ordered_index])

    return mask_data, info_data


def show_case(image: ndarray,
              expected_mask: ndarray,
              obtained_mask: ndarray,
              save_path: str):
    pyplot.figure(figsize=(10, 3), tight_layout=True)
    pyplot.subplot(1, 3, 1)
    pyplot.title("original image")
    pyplot.imshow(image)
    pyplot.axis("off")
    pyplot.subplot(1, 3, 2)
    pyplot.pcolormesh(arange(expected_mask.shape[1] + 1), arange(expected_mask.shape[0] + 1)[::-1], expected_mask,
                      cmap="rainbow", vmin=min(expected_mask), vmax=max(expected_mask))
    pyplot.axis("off")
    pyplot.subplot(1, 3, 3)
    pyplot.pcolormesh(arange(obtained_mask.shape[1] + 1), arange(obtained_mask.shape[0] + 1)[::-1], obtained_mask,
                      cmap="rainbow", vmin=min(obtained_mask), vmax=max(obtained_mask))
    pyplot.axis("off")

    if save_path is not None:
        pyplot.savefig(save_path, bbox_inches="tight")
    else:
        pyplot.show()
    pyplot.close()


def compare(expected_mask: ndarray,
            obtained_mask: ndarray) \
        -> dict:
    """
    Compare two mask of cell image.

    :param expected_mask: expected cell segmentation mask.
    :type expected_mask: ndarray
    :param obtained_mask: obtained cell segmentation mask.
    :type obtained_mask: ndarray

    :return: stats dictionary of expected and obtained cell segmentation mask.
    :rtype: dict
    """
    data = matching(y_true=expected_mask, y_pred=obtained_mask)
    return {"fp": data.fp, "tp": data.tp, "fn": data.fn,
            "precision": data.precision, "recall": data.recall, "accuracy": data.accuracy, "f1": data.f1,
            "criterion": data.criterion, "thresh": data.thresh, "n true": data.n_true, "n pred": data.n_pred,
            "mean true score": data.mean_true_score, "mean matched score": data.mean_matched_score,
            "panoptic quality": data.panoptic_quality}
