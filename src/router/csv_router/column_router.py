
import pandas as pd
from utils.registry import IRouter,IModelRegistry


class ColumnRouter(IRouter):
    def __init__(self, registry: IModelRegistry):
        self.registry = registry

    def route(self, df: pd.DataFrame) -> str:
        meta = self.registry.get_model_meta(df)
        return meta["name"]