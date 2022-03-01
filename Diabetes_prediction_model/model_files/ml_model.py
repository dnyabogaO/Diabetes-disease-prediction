##importing a few general use case libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_origin_cols(df):
    preproc_df = df.drop('Outcom', axis=1)
    return df

def num_pipeline_transformer(X):

    numerics = ['float64', 'int64']

    num_attrs = X.select_dtypes(include=numerics)

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])
    return num_attrs, num_pipeline


def pipeline_transformer(X):
    num_attrs, num_pipeline = num_pipeline_transformer(X)
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, list(num_attrs)),
    ])
    prepared_data = full_pipeline.fit_transform(X)
    return prepared_data


def predict_diab(config, model):
    if type(config) == dict:
        df = pd.DataFrame(config)
    else:
        df = config

    preproc_df = preprocess_origin_cols(df)
    prepared_df = pipeline_transformer(preproc_df)
    y_pred = model.predict(prepared_df)
    return y_pred