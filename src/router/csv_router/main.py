from .model_predictor import CSVModelPredictor
from .model_registry import ModelRegistry
from .column_router import ColumnRouter
from .router import RouterService

def master_predict(input_file):
    registry = ModelRegistry()
    router = ColumnRouter(registry)
    predictor = CSVModelPredictor(registry)

    service = RouterService(router, predictor)
    return service.predict_from_csv(input_file)
