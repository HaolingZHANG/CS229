from datetime import datetime
from matplotlib import pyplot
from numpy import ndarray, array, arange, zeros, repeat, min, max, sum, ceil, argmin, where
from os import listdir, path, mkdir
from PIL import Image
from stardist.matching import matching
from typing import Tuple


class Monitor(object):

    def __init__(self):
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


def load_train_pair(verbose: bool = False) \
        -> Tuple[list, list]:
    """
    Load train pairs from a folder.

    :param verbose: show the process.
    :type verbose: bool

    :return: train data pair.
    :rtype: list, list
    """
    folder_path = "../datasets/train/"
    value_1, value_2, indices, monitor = {}, {}, set(), Monitor()
    if verbose:
        print("load train images.")
        child_paths = list(listdir(folder_path))
        for process_index, child_path in enumerate(child_paths):
            image, info = array(Image.open(folder_path + child_path)), child_path[:-11]
            if "label" in child_path:
                value_1[info] = image
            else:
                image = repeat(image[:, :, None], 3, axis=-1) if len(image.shape) == 2 else image
                value_2[info] = image
            indices.add(info)
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
            image, info = array(Image.open(folder_path + child_path)), child_path[:-11]
            if "label" in child_path:
                value_1[info] = image
            else:
                image = repeat(image[:, :, None], 3, axis=-1) if len(image.shape) == 2 else image
                value_2[info] = image
            indices.add(info)

        mask_data, info_data = [], []
        for ordered_index in sorted(list(indices)):
            if ordered_index in value_1 and ordered_index in value_2:
                mask_data.append(value_1[ordered_index])
                info_data.append(value_2[ordered_index])

    return mask_data, info_data


def load_test_pair(verbose: bool = False) \
        -> Tuple[list, list]:
    """
    Load test pairs from a folder.

    :param verbose: show the process.
    :type verbose: bool

    :return: test data pair.
    :rtype: list, list
    """
    folder_path = "../datasets/test/"
    value_1, value_2, indices, monitor = {}, {}, set(), Monitor()
    if verbose:
        print("load test images.")
        chile_paths = list(listdir(folder_path))
        for process_index, child_path in enumerate(chile_paths):
            image, info = array(Image.open(folder_path + child_path)), child_path[:-13]
            if "label" in child_path:
                value_1[info] = image
            else:
                image = repeat(image[:, :, None], 3, axis=-1) if len(image.shape) == 2 else image
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
                image = repeat(image[:, :, None], 3, axis=-1) if len(image.shape) == 2 else image
                value_2[info] = image
            indices.add(info)

        mask_data, info_data = [], []
        for ordered_index in sorted(list(indices)):
            if ordered_index in value_1 and ordered_index in value_2:
                mask_data.append(value_1[ordered_index])
                info_data.append(value_2[ordered_index])

    return mask_data, info_data


def load_tune_pair(side: int,
                   overlap: float,
                   verbose: bool = False) \
        -> Tuple[list, list]:
    """
    Load fine-tune pairs from a folder.

    :param side: minimum weight and height in the images.
    :type side: int

    :param overlap: overlap size between image samples.
    :type overlap: float

    :param verbose: show the process.
    :type verbose: bool

    :return: tune data pair.
    :rtype: list, list
    """
    folder_path = "../datasets/tune/"
    value_1, value_2, minimum_size, indices, monitor = {}, {}, ceil(side * (1 + 2 * overlap)), set(), Monitor()
    if verbose:
        print("load tune images.")
        chile_paths = list(listdir(folder_path))
        for process_index, child_path in enumerate(chile_paths):
            image, info = Image.open(folder_path + child_path), child_path[:-11]
            if image.size[0] < minimum_size or image.size[1] < minimum_size:  # change the size if not applicable.
                change = max([minimum_size / image.size[0], minimum_size / image.size[1]])
                image = image.resize(size=(int(image.size[0] * change), int(image.size[1] * change)))
            if "label" in child_path:
                value_1[info] = array(image)
            else:
                image = array(image)
                image = repeat(image[:, :, None], 3, axis=-1) if len(image.shape) == 2 else image
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
            image, info = Image.open(folder_path + child_path), child_path[:-11]
            if image.size[0] < minimum_size or image.size[1] < minimum_size:  # change the size if not applicable.
                change = max([minimum_size / image.size[0], minimum_size / image.size[1]])
                image = image.resize(size=(int(image.size[0] * change), int(image.size[1] * change)))
            if "label" in child_path:
                value_1[info] = array(image)
            else:
                image = array(image)
                image = repeat(image[:, :, None], 3, axis=-1) if len(image.shape) == 2 else image
                value_2[info] = image
            indices.add(info)

        mask_data, info_data = [], []
        for ordered_index in sorted(list(indices)):
            if ordered_index in value_1 and ordered_index in value_2:
                mask_data.append(value_1[ordered_index])
                info_data.append(value_2[ordered_index])

    return mask_data, info_data


def batchize(labels: list,
             images: list,
             classes: list,
             batch_capacity: int,
             side: int,
             overlap: float,
             folder_path: str,
             verbose: bool = False) \
        -> int:
    """
    Batchize the image-mask pairs.

    :param labels: labels or masks of images.
    :type labels: list

    :param images: images for detecting.
    :type images: list

    :param classes: artificial classes of images.
    :type classes: list

    :param batch_capacity: pair number of sub-image and sub mask in each batch.
    :type batch_capacity: int

    :param side: side of images used for normalization.
    :type side: int

    :param overlap: overlap size between image samples.
    :type overlap: float

    :param folder_path: folder path to save batched images.
    :type folder_path: str

    :param verbose: show the process.
    :type verbose: bool

    :return: batch number.
    :rtype: int
    """
    if not path.exists(folder_path):
        mkdir(folder_path)
    else:
        return len(list(listdir(folder_path)))

    margin = int(side * overlap)
    if verbose:
        print("Count the number of sub-image with shape (%d, %d) in each class." % (side, side))
        identities, total_count, monitor = zeros(shape=(len(labels), 2), dtype=int), 0, Monitor()
        for index, label in enumerate(labels):
            for group_index, group in enumerate(classes):
                if index + 1 in group:
                    identities[index, 0] = group_index
                    break

            count, bottom_location = 0, 0
            for x_index in range(0, label.shape[0] - side + 1, int(side * (1 - overlap))):
                right_location = 0
                for y_index in range(0, label.shape[1] - side + 1, int(side * (1 - overlap))):
                    count, right_location = count + 1, y_index + side
                if label.shape[1] - right_location > margin:
                    count += 1
            if label.shape[0] - bottom_location > margin:
                for y_index in range(0, label.shape[1] - side + 1, side // 2):
                    count += 1
            identities[index, 1], total_count = count, total_count + count
            monitor(index + 1, len(labels), extra={"total samples": total_count})

        counter = {}
        for identity in identities:
            if identity[0] in counter:
                counter[identity[0]] += identity[1]
            else:
                counter[identity[0]] = identity[1]
        batch_sizes = zeros(shape=(len(classes),), dtype=int)
        for key, value in counter.items():
            batch_sizes[key] = int(value / float(total_count) * batch_capacity)
        if sum(batch_sizes) < batch_capacity:
            batch_sizes[argmin(batch_sizes)] += batch_capacity - sum(batch_sizes)
        print("each class in batch is:" + str(batch_sizes))

        total_group = int(total_count / batch_capacity)
        for group_index in range(total_group):
            mkdir(folder_path + str(group_index + 1).zfill(len(str(total_group))) + "/")

        print("split into different batches.")
        for class_index in range(len(classes)):
            single_identities, flag, group_index, count = where(identities[:, 0] == class_index)[0], False, 0, 0
            for identity in single_identities:
                label, image, bottom_location = labels[identity], images[identity], 0
                for x_index in range(0, label.shape[0] - side + 1, int(side * (1 - overlap))):
                    right_location = 0
                    for y_index in range(0, label.shape[1] - side + 1, int(side * (1 - overlap))):
                        sub_image = image[x_index: x_index + side, y_index: y_index + side]
                        sub_label = label[x_index: x_index + side, y_index: y_index + side]
                        if sub_label.shape != (1024, 1024):
                            print(1, sub_image.shape, sub_label.shape)
                        right_location = y_index + side
                        batch_path = folder_path + str(group_index + 1).zfill(len(str(total_group))) + "/"
                        index = len(list(listdir(batch_path))) // 2 + 1
                        image_path = batch_path + str(index).zfill(len(str(batch_capacity))) + ".image.tiff"
                        Image.fromarray(sub_image).save(image_path)
                        label_path = batch_path + str(index).zfill(len(str(batch_capacity))) + ".label.tiff"
                        Image.fromarray(sub_label).save(label_path)
                        count += 1
                        if count == batch_sizes[class_index]:
                            group_index, count = group_index + 1, 0
                        if group_index == total_group:
                            flag = True
                            break
                    if flag:
                        break
                    if label.shape[1] - right_location > margin:
                        sub_image = image[x_index: x_index + side, label.shape[1] - side:]
                        sub_label = label[x_index: x_index + side, label.shape[1] - side:]
                        if sub_label.shape != (1024, 1024):
                            print(2, sub_image.shape, sub_label.shape)
                        batch_path = folder_path + str(group_index + 1).zfill(len(str(total_group))) + "/"
                        index = len(list(listdir(batch_path))) // 2 + 1
                        image_path = batch_path + str(index).zfill(len(str(batch_capacity))) + ".image.tiff"
                        Image.fromarray(sub_image).save(image_path)
                        label_path = batch_path + str(index).zfill(len(str(batch_capacity))) + ".label.tiff"
                        Image.fromarray(sub_label).save(label_path)
                        count += 1
                        if count == batch_sizes[class_index]:
                            group_index, count = group_index + 1, 0
                        if group_index == total_group:
                            flag = True
                            break
                    if flag:
                        break
                if flag:
                    break
                if label.shape[0] - bottom_location > margin:
                    for y_index in range(0, label.shape[1] - side + 1, int(side * (1 - overlap))):
                        sub_image = image[max([label.shape[0] - side, 0]):, y_index: y_index + side]
                        sub_label = label[max([label.shape[0] - side, 0]):, y_index: y_index + side]
                        if sub_label.shape != (1024, 1024):
                            print(3, sub_image.shape, sub_label.shape)
                        batch_path = folder_path + str(group_index + 1).zfill(len(str(total_group))) + "/"
                        index = len(list(listdir(batch_path))) // 2 + 1
                        image_path = batch_path + str(index).zfill(len(str(batch_capacity))) + ".image.tiff"
                        Image.fromarray(sub_image).save(image_path)
                        label_path = batch_path + str(index).zfill(len(str(batch_capacity))) + ".label.tiff"
                        Image.fromarray(sub_label).save(label_path)
                        count += 1
                        if count == batch_sizes[class_index]:
                            group_index, count = group_index + 1, 0
                        if group_index == total_group:
                            flag = True
                            break
                    if flag:
                        break
                if flag:
                    break
            monitor(class_index + 1, len(classes))
    else:
        identities, total_count = zeros(shape=(len(labels), 2), dtype=int), 0
        for index, label in enumerate(labels):
            for group_index, group in enumerate(classes):
                if index + 1 in group:
                    identities[index, 0] = group_index
                    break

            count, bottom_location = 0, 0
            for x_index in range(0, label.shape[0] - side + 1, int(side * (1 - overlap))):
                right_location = 0
                for y_index in range(0, label.shape[1] - side + 1, int(side * (1 - overlap))):
                    count, right_location = count + 1, y_index + side
                if label.shape[1] - right_location > margin:
                    count += 1
            if label.shape[0] - bottom_location > margin:
                for y_index in range(0, label.shape[1] - side + 1, side // 2):
                    count += 1
            identities[index, 1], total_count = count, total_count + count

        counter = {}
        for identity in identities:
            if identity[0] in counter:
                counter[identity[0]] += identity[1]
            else:
                counter[identity[0]] = identity[1]
        batch_sizes = zeros(shape=(len(classes),), dtype=int)
        for key, value in counter.items():
            batch_sizes[key] = int(value / float(total_count) * batch_capacity)
        if sum(batch_sizes) < batch_capacity:
            batch_sizes[argmin(batch_sizes)] += batch_capacity - sum(batch_sizes)

        total_group = int(total_count / batch_capacity)
        for group_index in range(total_group):
            mkdir(folder_path + str(group_index + 1).zfill(len(str(total_group))) + "/")

        for class_index in range(len(classes)):
            single_identities, flag, group_index, count = where(identities[:, 0] == class_index)[0], False, 0, 0
            for identity in single_identities:
                label, image, bottom_location = labels[identity], images[identity], 0
                for x_index in range(0, label.shape[0] - side + 1, int(side * (1 - overlap))):
                    right_location = 0
                    for y_index in range(0, label.shape[1] - side + 1, int(side * (1 - overlap))):
                        sub_image = image[x_index: x_index + side, y_index: y_index + side]
                        sub_label = label[x_index: x_index + side, y_index: y_index + side]
                        right_location = y_index + side
                        batch_path = folder_path + str(group_index + 1).zfill(len(str(total_group))) + "/"
                        index = len(list(listdir(batch_path))) // 2 + 1
                        image_path = batch_path + str(index).zfill(len(str(batch_capacity))) + ".image.tiff"
                        Image.fromarray(sub_image).save(image_path)
                        label_path = batch_path + str(index).zfill(len(str(batch_capacity))) + ".label.tiff"
                        Image.fromarray(sub_label).save(label_path)
                        count += 1
                        if count == batch_sizes[class_index]:
                            group_index, count = group_index + 1, 0
                        if group_index == total_group:
                            flag = True
                            break
                    if flag:
                        break
                    if label.shape[1] - right_location > margin:
                        sub_image = image[x_index: x_index + side, label.shape[1] - side:]
                        sub_label = label[x_index: x_index + side, label.shape[1] - side:]

                        batch_path = folder_path + str(group_index + 1).zfill(len(str(total_group))) + "/"
                        index = len(list(listdir(batch_path))) // 2 + 1
                        image_path = batch_path + str(index).zfill(len(str(batch_capacity))) + ".image.tiff"
                        Image.fromarray(sub_image).save(image_path)
                        label_path = batch_path + str(index).zfill(len(str(batch_capacity))) + ".label.tiff"
                        Image.fromarray(sub_label).save(label_path)
                        count += 1
                        if count == batch_sizes[class_index]:
                            group_index, count = group_index + 1, 0
                        if group_index == total_group:
                            flag = True
                            break
                    if flag:
                        break
                if flag:
                    break
                if label.shape[0] - bottom_location > margin:
                    for y_index in range(0, label.shape[1] - side + 1, int(side * (1 - overlap))):
                        sub_image = image[label.shape[0] - side:, y_index: y_index + side]
                        sub_label = label[label.shape[0] - side:, y_index: y_index + side]

                        batch_path = folder_path + str(group_index + 1).zfill(len(str(total_group))) + "/"
                        index = len(list(listdir(batch_path))) // 2 + 1
                        image_path = batch_path + str(index).zfill(len(str(batch_capacity))) + ".image.tiff"
                        Image.fromarray(sub_image).save(image_path)
                        label_path = batch_path + str(index).zfill(len(str(batch_capacity))) + ".label.tiff"
                        Image.fromarray(sub_label).save(label_path)
                        count += 1
                        if count == batch_sizes[class_index]:
                            group_index, count = group_index + 1, 0
                        if group_index == total_group:
                            flag = True
                            break
                    if flag:
                        break
                if flag:
                    break

    return total_group


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
