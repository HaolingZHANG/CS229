from cellpose import models
from numpy import ndarray, array, zeros, where, uint8, ndenumerate
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from skimage import transform
from torch import Tensor, nn, no_grad, as_tensor, cuda, sigmoid, float32
from typing import Union


class BaselineModel:

    def __init__(self):
        self.model = models.CellposeModel(model_type="cyto2", pretrained_model="../models/")
        self.channels, self.flow_threshold = [0, 0], 0.4

    def __call__(self,
                 image: ndarray) \
            -> ndarray:
        return self.model.eval(image, channels=self.channels, flow_threshold=self.flow_threshold)[0]


class ComparedModel:

    def __init__(self,
                 model_type: str):
        device = "cuda:0" if cuda.is_available() else "cpu"
        if model_type == "medsam":
            sam = sam_model_registry["vit_b"](checkpoint="../models/medsam_vit_b.pth").to(device)
        elif model_type == "vit_b":
            sam = sam_model_registry[model_type](checkpoint="../models/sam_vit_b_01ec64.pth").to(device)
        elif model_type == "vit_l":
            sam = sam_model_registry[model_type](checkpoint="../models/sam_vit_l_0b3195.pth").to(device)
        else:
            sam = sam_model_registry[model_type](checkpoint="../models/sam_vit_h_4b8939.pth").to(device)
        self.model = SamAutomaticMaskGenerator(sam)

    def __call__(self,
                 image: ndarray,
                 structure_changed: bool = False) \
            -> ndarray:
        original_size, changed = image.shape, False
        if image.shape[0] > 1024 or image.shape[1] > 1024:
            image = transform.resize(image, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True)
            image = image.astype(uint8)
            changed = True

        mask = zeros(image.shape[:2], dtype=int)
        for index, mask_info in enumerate(self.model.generate(image)):
            mask[where(mask_info["segmentation"])] = index + 1

        if changed:
            mask = transform.resize(mask, original_size[:2], order=3, preserve_range=True, anti_aliasing=True)
            mask = mask.astype(uint8)

        if structure_changed:
            data = []
            for (x, y), mask_index in ndenumerate(mask):
                if mask_index > 0:
                    data.append([x, y, mask_index])
            return array(data)

        else:
            return mask


class FineTunedModel(nn.Module):

    def __init__(self,
                 model_type: str,
                 fine_tune: bool):
        super(FineTunedModel, self).__init__()
        assert model_type in ["vit_b", "vit_l", "vit_h"]
        self.device = "cuda:0" if cuda.is_available() else "cpu"
        if model_type == "vit_b":
            self.sam = sam_model_registry[model_type](checkpoint="../models/sam_vit_b_01ec64.pth").to(self.device)
        elif model_type == "vit_l":
            self.sam = sam_model_registry[model_type](checkpoint="../models/sam_vit_l_0b3195.pth").to(self.device)
        else:
            self.sam = sam_model_registry[model_type](checkpoint="../models/sam_vit_h_4b8939.pth").to(self.device)
        self.fine_tune = fine_tune

    def forward(self,
                image: Union[Tensor, ndarray]) \
            -> Union[Tensor, ndarray]:
        if self.fine_tune:
            assert isinstance(image, Tensor)
        else:
            assert isinstance(image, ndarray)

        if self.fine_tune:
            # do not compute gradients for image encoder and prompt encoder.
            with no_grad():
                image_embedding = self.sam.image_encoder(image)
                boxes = as_tensor(array([[[0, 0, image.shape[2], image.shape[3]]]]), dtype=float32, device=self.device)
                sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(points=None, boxes=boxes, masks=None)

            # make sure we only compute gradients for mask decoder.
            logits = self.sam.mask_decoder(image_embeddings=image_embedding,
                                           image_pe=self.sam.prompt_encoder.get_dense_pe(),
                                           sparse_prompt_embeddings=sparse_embeddings,
                                           dense_prompt_embeddings=dense_embeddings,
                                           multimask_output=True)[0]

            masks = nn.functional.interpolate(sigmoid(logits), size=(image.shape[2], image.shape[3]),
                                              mode="bilinear", align_corners=False)
            return masks
        else:
            mask, count = zeros(image.shape[:2], dtype=int), 0

            for index, mask_info in enumerate(SamAutomaticMaskGenerator(self.sam).generate(image)):
                mask[where(mask_info["segmentation"])] = index + 1
                count += 1

            return mask
