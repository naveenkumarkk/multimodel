from torchvision import datasets

class DatasetInfo:
    def __init__(self, path: str):
        self.dataset = datasets.ImageFolder(path)
        self.classes = self.dataset.classes
        self.num_classes = len(self.classes)
