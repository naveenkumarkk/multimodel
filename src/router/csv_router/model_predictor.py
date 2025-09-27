
import joblib
import pandas as pd
from typing import List, Any
from utils.registry import IModelRegistry,IPredictor
from utils.registry import IModelRegistry

class CSVModelPredictor(IPredictor):
    def __init__(self, registry: IModelRegistry):
        self.registry = registry

    def predict(self, df: pd.DataFrame) -> List[Any]:
        meta = self.registry.get_model_meta(df)
        model = joblib.load(meta["path"])
        features = df[meta["columns"]]
        features = features.select_dtypes(include=["number"])
        return model.predict(features)
