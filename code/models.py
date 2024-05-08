from cellpose import models
from numpy import ndarray, array, repeat, clip, min, max, uint8
from segment_anything import sam_model_registry
from skimage import transform
from torch import tensor, as_tensor, nn, cuda, no_grad, sigmoid


class BaselineModel:

    def __init__(self):
        self.model = models.CellposeModel(model_type="cyto2", pretrained_model="../models/")
        self.channels, self.flow_threshold = [0, 0], 0.4

    def __call__(self,
                 image: ndarray) \
            -> ndarray:
        return self.model.eval(image, diameter=None, channels=self.channels,
                               flow_threshold=self.flow_threshold, do_3D=False)[0]


class ComparedModel:

    def __init__(self):
        self.device = "cuda:0" if cuda.is_available() else "cpu"
        self.model = sam_model_registry["vit_b"](checkpoint="../models/medsam_vit_b.pth")
        self.model.to(self.device)
        self.model.eval()

    def __call__(self,
                 image: ndarray) \
            -> ndarray:
        new_image = repeat(image[:, :, None], 3, axis=-1) if len(image.shape) == 2 else image
        h, w, _ = new_image.shape
        image_1024 = transform.resize(new_image, (1024, 1024), order=3,
                                      preserve_range=True, anti_aliasing=True).astype(uint8)
        numerator = image_1024 - min(image_1024)
        denominator = clip(max(image_1024) - min(image_1024), a_min=1e-8, a_max=None)
        image_1024 = numerator / denominator
        image_1024 = tensor(image_1024).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        with no_grad():
            image_embedding = self.model .image_encoder(image_1024)
        box = as_tensor(array([[0.0, 0.0, 1024.0, 1024.0]]), device=self.device)
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=None, boxes=box, masks=None)
        low_res_logits, _ = self.model.mask_decoder(image_embeddings=image_embedding,
                                                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                                                    sparse_prompt_embeddings=sparse_embeddings,
                                                    dense_prompt_embeddings=dense_embeddings,
                                                    multimask_output=False)
        low_res_pred = sigmoid(low_res_logits)
        low_res_pred = nn.functional.interpolate(low_res_pred, size=(h, w), mode="bilinear", align_corners=False)
        low_res_pred = low_res_pred.cpu().squeeze().detach().numpy()
        mask = (low_res_pred > 0.5).astype(uint8)

        return mask


class RetrainedModel(nn.Module):

    pass
