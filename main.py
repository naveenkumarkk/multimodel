import os
from pathlib import Path
from utils.registry import IRouter,IPredictor
from utils.utils import BASE_DIR
from PIL import Image
import pandas as pd
from src.router.csv_router.model_registry import ModelRegistry
from src.router.csv_router.column_router import ColumnRouter
from src.router.csv_router.model_predictor import CSVModelPredictor
from src.router.image_router.main import ImageRouterService
from src.experts.cerevical.preprocess import PreprocessCervical

def abspath(*paths):
    return os.path.join(BASE_DIR, *paths)

ROUTER_MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "src/models/router_model.pth"))
class RouterService:
    def __init__(self, csv_router: IRouter, csv_predictor: IPredictor,image_predictor):
        self.csv_router = csv_router
        self.csv_predictor = csv_predictor
        self.image_predictor = image_predictor

    def predict(self, file_path: str):
        ext = Path(file_path).suffix.lower()

        if ext == ".csv":
            df = pd.read_csv(file_path,encoding="cp1252")
            model_name = self.csv_router.route(df)
            print(f"[CSV] {file_path} → routed to model: {model_name}")
            return self.csv_predictor.predict(df)

        elif ext in [".jpg", ".jpeg", ".png"]:
            
            return self.image_predictor.predict(file_path)

        else:
            print(f"[SKIP] Unsupported file: {file_path}")
            return None

    def run_test_folder(self, test_folder: str):
        results = {}
        for filename in os.listdir(test_folder):
            file_path = os.path.join(test_folder, filename)
            
            try:
                preds = self.predict(file_path)
                results[filename] = preds
            except Exception as e:
                results[filename] = f"{e}"
                
        return results

if __name__ == "__main__":
    registry = ModelRegistry()
    csv_router = ColumnRouter(registry)
    csv_predictor = CSVModelPredictor(registry)
    
    image_predictor = ImageRouterService(ROUTER_MODEL_PATH)

    service = RouterService(csv_router, csv_predictor, image_predictor)

    test_folder = "data/test_data"
    results = service.run_test_folder(test_folder)

    print("\n=== RESULTS ===")
    for fname, pred in results.items():
        print(f"{fname}: {pred}")
