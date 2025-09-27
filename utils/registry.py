from typing import Dict, List, Any
import abc
import pandas as pd

class IModelRegistry(abc.ABC):
    @abc.abstractmethod
    def get_model_meta(self, df: pd.DataFrame) -> Dict[str, Any]:
        pass


class IRouter(abc.ABC):
    @abc.abstractmethod
    def route(self, df: pd.DataFrame) -> str:
        pass


class IPredictor(abc.ABC):
    @abc.abstractmethod
    def predict(self, df: pd.DataFrame) -> List[Any]:
        pass
