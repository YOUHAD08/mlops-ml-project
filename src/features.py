# src/features.py
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

def _clip(X):
    """Clip extreme values to reduce outlier impact"""
    return X.clip(-3, 3)

def build_numeric_preprocess():
    """
    Improved preprocessing pipeline:
    - median imputation
    - standardization
    - clipping outliers
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clip", FunctionTransformer(_clip)),
    ])