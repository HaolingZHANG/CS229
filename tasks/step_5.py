from code import Monitor, FineTunedModel
from monai.losses import DiceCELoss
from numpy import array, expand_dims, transpose, inf
from os import path, mkdir, remove
from PIL import Image
from torch import tensor, nn, save, cuda, optim, float32
from typing import Tuple

monitor, device = Monitor(), "cuda:0" if cuda.is_available() else "cpu"


def load_dataset(batch_index: int,
                 batch_number: int,
                 batch_capacity: int) -> Tuple[list, list]:
    images, labels, batch_path = [], [], "../outputs/task3/" + str(batch_index).zfill(len(str(batch_number))) + "/"
    for index in range(1, batch_capacity + 1):
        image_path = batch_path + str(index).zfill(len(str(batch_capacity))) + ".image.tiff"
        image = array(Image.open(image_path).convert("RGB"))
        image = expand_dims(image, axis=0)
        image = transpose(image, (0, 3, 1, 2))
        images.append(tensor(image, dtype=float32, device=device))
        label_path = batch_path + str(index).zfill(len(str(batch_capacity))) + ".label.tiff"
        label = array(Image.open(label_path), dtype=float)
        labels.append(tensor(label, dtype=float32, device=device))

    return images, labels


def task(batch_number: int,
         batch_capacity: int,
         maximum_iteration: int):

    if not path.exists("../outputs/task5/"):
        mkdir("../outputs/task5/")
        model = FineTunedModel(model_type="vit_b", fine_tune=True)

        optimizer = optim.Adam(model.sam.mask_decoder.parameters(), lr=1e-5)
        seg_loss, ce_loss = DiceCELoss(sigmoid=True, squared_pred=True), nn.BCEWithLogitsLoss()
        for batch_index in range(1, batch_number + 1):
            print("use batch %d." % batch_index)
            images, expected_masks = load_dataset(batch_index, batch_number, batch_capacity)
            best_loss, last_saved_path = inf, ""
            for epoch in range(maximum_iteration):
                epoch_loss = 0.0
                for image, expected_mask in zip(images, expected_masks):
                    optimizer.zero_grad()
                    obtained_mask = model(image)
                    loss = seg_loss(obtained_mask, expected_mask) + ce_loss(obtained_mask, expected_mask)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                monitor(epoch + 1, maximum_iteration, extra={"current loss": epoch_loss, "saved loss": best_loss})

                if epoch_loss < best_loss:
                    if len(last_saved_path) > 0:
                        remove(last_saved_path)

                    batch_path = "../outputs/task5/" + str(batch_index).zfill(len(str(batch_number))) + "."
                    info = str(epoch).zfill(len(str(maximum_iteration))) + "." + "%.3f" % epoch_loss + ".pth"
                    last_saved_path = batch_path + info
                    save(model.state_dict(), last_saved_path)
                    best_loss = epoch_loss


if __name__ == "__main__":
    task(batch_number=40, batch_capacity=16, maximum_iteration=1000)
