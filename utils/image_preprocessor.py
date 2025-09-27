from torchvision import transforms
from PIL import Image

class ImagePreprocessor:
    def __init__(self, size=(224, 224)):
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def process(self, img_path: str):
        img = Image.open(img_path).convert("RGB")
        return self.transform(img)
