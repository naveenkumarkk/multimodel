import torch
from torchvision import datasets
from .predictor import Predictor

class ExpertLoader:
    def __init__(self, experts_registry):
        self.experts_registry = experts_registry

    def load_expert(self, expert_name: str):
        if expert_name not in self.experts_registry:
            raise ValueError(f"Unknown expert: {expert_name}")

        expert_info = self.experts_registry[expert_name]

        dataset = datasets.ImageFolder(expert_info["dataset_path"])
        class_names = dataset.classes
        num_classes = len(class_names)

        expert_model = expert_info["model_class"](num_classes=num_classes)
        state_dict = torch.load(expert_info["model_path"], map_location="cpu", weights_only=False)
        expert_model.model.load_state_dict(state_dict)
        expert_model.to("cpu")

        predictor = Predictor(expert_model, class_names)
        return predictor
