import pandas as pd
# from ....utils.registry import IModelRegistry
from typing import Dict, List, Any
import os
from utils.utils import BASE_DIR,abspath

class ModelRegistry():
    """Holds model paths and their expected columns."""

    MODEL_REGISTRY = {
        "breast": {
            "path": abspath("src/models", "knn_breast_cancer_model.pkl"),
            "columns": [
                "Clump_thickness", "Uniformity_Cell_Size", "Uniformity_Cell_Shape",
                "Marginal_Adhesion", "Single_Epithelial_Cell_Size", "Bare_Nuclei",
                "Bland_Chromatin", "Normal_Nucleoli", "Mitoses"
            ],
        },
        "cancer_classification": {
            "path":abspath("src/models","svm_cancer_classification_model.pkl"), 
            "columns": [
                "mean radius", "mean texture", "mean perimeter", "mean area",
                "mean smoothness", "mean compactness", "mean concavity",
                "mean concave points", "mean symmetry", "mean fractal dimension",
                "radius error", "texture error", "perimeter error", "area error",
                "smoothness error", "compactness error", "concavity error",
                "concave points error", "symmetry error", "fractal dimension error",
                "worst radius", "worst texture", "worst perimeter", "worst area",
                "worst smoothness", "worst compactness", "worst concavity",
                "worst concave points", "worst symmetry", "worst fractal dimension"
            ],
        },
        "cervical_classification": {
            "path":abspath("src/models","cervical_pipeline.pkl"),
            "columns": [
                "Age", "Number of sexual partners", "First sexual intercourse",
                "Num of pregnancies", "Smokes", "Smokes (years)", "Smokes (packs/year)",
                "Hormonal Contraceptives", "Hormonal Contraceptives (years)",
                "IUD", "IUD (years)", "STDs", "STDs (number)", "STDs:condylomatosis",
                "STDs:cervical condylomatosis", "STDs:vaginal condylomatosis",
                "STDs:vulvo-perineal condylomatosis", "STDs:syphilis",
                "STDs:pelvic inflammatory disease", "STDs:genital herpes",
                "STDs:molluscum contagiosum", "STDs:AIDS", "STDs:HIV", "STDs:Hepatitis B",
                "STDs:HPV", "STDs: Number of diagnosis", "Dx:CIN", "Dx:HPV", "Dx",
                "Hinselmann", "Schiller", "Citology","Biopsy"
            ],
            
        },
        "text_classification": {
            "path":abspath("src/models","cancer_text_classification.pkl"),
            "columns": ["reviews"],
        },
    }

    def get_model_meta(self, df: pd.DataFrame) -> Dict[str, Any]:
        cols = set(df.columns)
        for name, meta in self.MODEL_REGISTRY.items():
            if set(meta["columns"]).issubset(cols):
                return {"name": name, **meta}
        raise ValueError(f"No matching model found for columns: {df.columns.tolist()}")

