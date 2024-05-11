from code import Monitor, load_test_pair, compare
from numpy import ndarray, zeros, where, uint8
from os import path, mkdir
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from skimage import transform
from tasks import save_data
from torch import cuda

monitor, device = Monitor(), "cuda:0" if cuda.is_available() else "cpu"


def obtain_mask(model: SamAutomaticMaskGenerator,
                image: ndarray) \
        -> ndarray:
    mask, count = zeros(image.shape[:2], dtype=int), 0
    for index, mask_info in enumerate(model.generate(image)):
        mask[where(mask_info["segmentation"])] = index + 1
        count += 1
    return mask


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

        print("run the test for baseline SAM.")
        sam = sam_model_registry["vit_b"](checkpoint="../models/sam_vit_b_01ec64.pth").to(device)
        model = SamAutomaticMaskGenerator(sam)
        for process_index, (expected_mask, image) in enumerate(zip(mask_data, info_data)):
            save_path = "../outputs/task4/b." + str(process_index + 1).zfill(len(str(len(info_data)))) + ".pkl"
            if not path.exists(save_path):
                original_size, changed = image.shape, False
                if image.shape[0] > 1024 or image.shape[1] > 1024:
                    image = transform.resize(image, (1024, 1024), order=3,
                                             preserve_range=True, anti_aliasing=True).astype(uint8)
                    changed = True
                obtained_mask = obtain_mask(model=model, image=image)
                if changed:
                    obtained_mask = transform.resize(obtained_mask, original_size[:2], order=3,
                                                     preserve_range=True, anti_aliasing=True).astype(uint8)
                stats = compare(expected_mask=expected_mask, obtained_mask=obtained_mask)
                save_data(save_path=save_path, information=stats)
            monitor(process_index + 1, len(info_data))

        print("run the test for low resolution SAM.")
        sam = sam_model_registry["vit_l"](checkpoint="../models/sam_vit_l_0b3195.pth").to(device)
        model = SamAutomaticMaskGenerator(sam)
        for process_index, (expected_mask, image) in enumerate(zip(mask_data, info_data)):
            save_path = "../outputs/task4/l." + str(process_index + 1).zfill(len(str(len(info_data)))) + ".pkl"
            if not path.exists(save_path):
                original_size, changed = image.shape, False
                if image.shape[0] > 1024 or image.shape[1] > 1024:
                    image = transform.resize(image, (1024, 1024), order=3,
                                             preserve_range=True, anti_aliasing=True).astype(uint8)
                    changed = True
                obtained_mask = obtain_mask(model=model, image=image)
                if changed:
                    obtained_mask = transform.resize(obtained_mask, original_size[:2], order=3,
                                                     preserve_range=True, anti_aliasing=True).astype(uint8)
                stats = compare(expected_mask=expected_mask, obtained_mask=obtained_mask)
                save_data(save_path=save_path, information=stats)
            monitor(process_index + 1, len(info_data))

        print("run the test for high resolution SAM.")
        sam = sam_model_registry["vit_h"](checkpoint="../models/sam_vit_h_4b8939.pth").to(device)
        model = SamAutomaticMaskGenerator(sam)
        for process_index, (expected_mask, image) in enumerate(zip(mask_data, info_data)):
            save_path = "../outputs/task4/h." + str(process_index + 1).zfill(len(str(len(info_data)))) + ".pkl"
            if not path.exists(save_path):
                original_size, changed = image.shape, False
                if image.shape[0] > 1024 or image.shape[1] > 1024:
                    image = transform.resize(image, (1024, 1024), order=3,
                                             preserve_range=True, anti_aliasing=True).astype(uint8)
                    changed = True
                obtained_mask = obtain_mask(model=model, image=image)
                if changed:
                    obtained_mask = transform.resize(obtained_mask, original_size[:2], order=3,
                                                     preserve_range=True, anti_aliasing=True).astype(uint8)
                stats = compare(expected_mask=expected_mask, obtained_mask=obtained_mask)
                save_data(save_path=save_path, information=stats)
            monitor(process_index + 1, len(info_data))


if __name__ == "__main__":
    task()
