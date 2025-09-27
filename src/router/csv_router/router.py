
import pandas as pd
from typing import  List, Any
from ....utils.registry import IRouter,IPredictor

class RouterService:
    def __init__(self, router: IRouter, predictor: IPredictor):
        self.router = router
        self.predictor = predictor

    def predict_from_csv(self, csv_file: str) -> List[Any]:
        df = pd.read_csv(csv_file)
        model_name = self.router.route(df)
        print(f"Routed to model: {model_name}")
        return self.predictor.predict(df)

