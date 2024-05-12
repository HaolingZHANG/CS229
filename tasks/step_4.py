from code import Monitor, ComparedModel, load_test_pair, compare
from os import path, mkdir
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

    print("run the test for baseline SAM.")
    model = ComparedModel(model_type="vit_b")
    for process_index, (expected_mask, image) in enumerate(zip(mask_data, info_data)):
        save_path = "../outputs/task4/b." + str(process_index + 1).zfill(len(str(len(info_data)))) + ".pkl"
        if not path.exists(save_path):
            stats = compare(expected_mask=expected_mask, obtained_mask=model(image))
            save_data(save_path=save_path, information=stats)
        else:
            stats = load_data(load_path=save_path)
        monitor(process_index + 1, len(info_data), extra={"F1 score": stats["f1"]})

    print("run the test for low resolution SAM.")
    model = ComparedModel(model_type="vit_l")
    for process_index, (expected_mask, image) in enumerate(zip(mask_data, info_data)):
        save_path = "../outputs/task4/l." + str(process_index + 1).zfill(len(str(len(info_data)))) + ".pkl"
        if not path.exists(save_path):
            stats = compare(expected_mask=expected_mask, obtained_mask=model(image))
            save_data(save_path=save_path, information=stats)
        else:
            stats = load_data(load_path=save_path)
        monitor(process_index + 1, len(info_data), extra={"F1 score": stats})

    print("run the test for high resolution SAM.")
    model = ComparedModel(model_type="vit_h")
    for process_index, (expected_mask, image) in enumerate(zip(mask_data, info_data)):
        save_path = "../outputs/task4/h." + str(process_index + 1).zfill(len(str(len(info_data)))) + ".pkl"
        if not path.exists(save_path):
            stats = compare(expected_mask=expected_mask, obtained_mask=model(image))
            save_data(save_path=save_path, information=stats)
        else:
            stats = load_data(load_path=save_path)
        monitor(process_index + 1, len(info_data), extra={"F1 score": stats})


if __name__ == "__main__":
    task()
