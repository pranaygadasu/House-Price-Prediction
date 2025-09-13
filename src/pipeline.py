import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# --- Feature engineering helpers ---
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if set(['TotalBsmtSF','1stFlrSF','2ndFlrSF']).issubset(out.columns):
        out['TotalSF'] = out[['TotalBsmtSF','1stFlrSF','2ndFlrSF']].sum(axis=1)
    if 'YearBuilt' in out.columns and 'YrSold' in out.columns:
        out['HouseAge'] = out['YrSold'] - out['YearBuilt']
    if 'SalePrice' in out.columns and 'SalePriceLog' not in out.columns:
        out['SalePriceLog'] = np.log1p(out['SalePrice'])
    return out

def build_preprocessor(df: pd.DataFrame):
    y_col = 'SalePriceLog' if 'SalePriceLog' in df.columns else 'SalePrice'
    feature_cols = [c for c in df.columns if c not in ['SalePrice','SalePriceLog','Id']]
    num_cols = df[feature_cols].select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols)
    ])
    return preprocessor, feature_cols, y_col
