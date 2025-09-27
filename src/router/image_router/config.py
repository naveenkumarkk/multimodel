from ...experts.brain.brain_expert import BrainClassifier
from ...experts.kidney.kidney_expert import KidneyClassifier
from ...experts.skin.skin_expert import SkinClassifier
from ...experts.pancrease.pancrease_expert import PancreaseClassifier
import os
from utils.utils import BASE_DIR,abspath


ROUTER_MODEL_PATH = os.path.join(BASE_DIR, "multimodel/src/models/router_model.pth")

EXPERTS = {
    "brain": {
        "model_class": BrainClassifier,
        "dataset_path": abspath("data", "Cancer_Dataset", "train", "brain"),
        "model_path": abspath("src/models", "brain_model.pth")
    },
    "kidney": {
        "model_class": KidneyClassifier,
        "dataset_path": abspath("data", "Cancer_Dataset", "train", "kidney"),
        "model_path": abspath("src/models", "kidney_model.pth")
    },
    "pancrease": {
        "model_class": PancreaseClassifier,
        "dataset_path": abspath("data", "Cancer_Dataset", "train", "pancrease"),
        "model_path": abspath("src/models", "pancrease_model.pth")
    },
    "skin": {
        "model_class": SkinClassifier,
        "dataset_path": abspath("data", "Cancer_Dataset", "train", "skin"),
        "model_path": abspath("src/models", "skin_model.pth")
    },
}
