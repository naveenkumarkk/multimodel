import os
import torch
import torchvision

from utils.dataset_info import DatasetInfo
from utils.utils import BASE_DIR
from utils.image_preprocessor import ImagePreprocessor
from .router_model import RouterModel
from .router import Router

from .expert_loader import ExpertLoader

from src.router.image_router.config import EXPERTS

SAFE_GLOBALS = [
    torchvision.models.resnet.ResNet,
    torch.nn.modules.conv.Conv2d,
    torch.nn.modules.batchnorm.BatchNorm2d
]
torch.serialization.add_safe_globals(SAFE_GLOBALS)


def abspath(*paths):
    return os.path.join(BASE_DIR, *paths)

class ImageRouterService:
    def __init__(self, router_model_path: str):
        dataset_info = DatasetInfo(abspath("data","specific_cancer_dataset", "train"))

        self.router_model = RouterModel(num_classes=dataset_info.num_classes)
        state_dict = torch.load(router_model_path, map_location="cpu", weights_only=False)
        self.router_model.load_state_dict(state_dict)
        self.router_model.to("cpu")

        self.router = Router(self.router_model, dataset_info.classes)
        self.preprocessor = ImagePreprocessor()
        self.expert_loader = ExpertLoader(EXPERTS)

    def predict(self, img_path: str, threshold: float = 0.7):
        img_tensor = self.preprocessor.process(img_path)
        pred_class, confidence = self.router.predict(img_tensor, threshold=threshold)
        predictor = self.expert_loader.load_expert(pred_class)
        sub_pred_class, sub_confidence = predictor.predict(img_tensor, threshold)

        return {
            "router_class": pred_class,
            "router_confidence": confidence,
            "expert_prediction": sub_pred_class,
            "expert_confidence": sub_confidence
        }
