from code import Monitor, FineTunedModel, load_test_pair, compare
from os import path, mkdir, listdir
from tasks import save_data
from torch import load, cuda


monitor, device = Monitor(), "cuda:0" if cuda.is_available() else "cpu"


def task(batch_number: int):
    if not path.exists("../outputs/task6/"):
        mkdir("../outputs/task6/")

        total_paths = []
        for child_path in listdir("../outputs/task5/"):
            total_paths.append("../outputs/task5/" + child_path)

        mask_data, info_data = load_test_pair(verbose=True)

        print("run the test.")
        for batch_index in range(1, batch_number + 1):
            batch_info = str(batch_index).zfill(len(str(batch_number)))
            pattern, model_path = "/" + batch_info + ".", None
            for total_path in total_paths:
                if pattern in total_path:
                    model_path = total_path
                    break
            model = FineTunedModel(model_type="vit_b", fine_tune=False)
            model.load_state_dict(load(model_path))

            print("run the fine-tuned model with batch %d." % batch_index)
            for process_index, (expected_mask, image) in enumerate(zip(mask_data, info_data)):
                sample_info = str(process_index + 1).zfill(len(str(len(info_data))))
                save_path = "../outputs/task6/" + batch_info + "." + sample_info + ".pkl"
                if not path.exists(save_path):
                    obtained_mask = model(image=image)
                    stats = compare(expected_mask=expected_mask, obtained_mask=obtained_mask)
                    save_data(save_path=save_path, information=stats)


if __name__ == "__main__":
    task(batch_number=40)
