import torch
import torch.nn as nn

class Router:
    def __init__(self, model: nn.Module, class_names: list, device=None):
        self.model = model
        self.class_names = class_names
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, img_tensor, threshold=0.7):
        with torch.no_grad():
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, dim=1)
            if conf.item() < threshold:
                return "unspecified", conf.item()
            else:
                return self.class_names[pred.item()], conf.item()
