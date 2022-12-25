from feature_engine.encoding import OrdinalEncoder, RareLabelEncoder
from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer,
)
from feature_engine.selection import DropFeatures
from feature_engine.transformation import LogTransformer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    QuantileTransformer,
    MinMaxScaler,
    OneHotEncoder,
    PowerTransformer,
)
import sys, os
import joblib
import pandas as pd
import algorithm.preprocessing.preprocessors as preprocessors


PREPROCESSOR_FNAME = "preprocessor.save"


"""

PRE-POCESSING STEPS =====>

=========== initial pre-processing ========
- Filter out 'info' variables

=========== for categorical variables ========
- Handle missing values in categorical:
    - When missing values are frequent, then impute with 'missing' label
    - When missing values are rare, then impute with most frequent
- Group rare labels to reduce number of categories 
# NOT DONE =>>> - Categorical variables: convert categories to ordinal scale by correlating to target
- One hot encode categorical variables

=========== for numerical variables ========
- Add binary column to represent 'missing' flag for missing values
- Impute missing values with mean of non-missing
- MinMax scale variables prior to yeo-johnson transformation
- Use Yeo-Johnson transformation to get (close to) gaussian dist. 
- Standard scale data after yeo-johnson

=========== for target variable ========
- Use Yeo-Johnson transformation to get (close to) gaussian dist. 
- Standard scale target data after yeo-johnson
===============================================
"""


def get_preprocess_pipeline(pp_params, model_cfg):

    pp_step_names = model_cfg["pp_params"]["pp_step_names"]

    pipe_steps = []

    # ===== KEEP ONLY COLUMNS WE USE   =====
    pipe_steps.append(
        (
            pp_step_names["COLUMN_SELECTOR"],
            preprocessors.ColumnSelector(columns=pp_params["retained_vars"]),
        )
    )

    # ===============================================================
    # ===== NUMERICAL VARIABLES =====

    # Transform numerical variables - standard
    if len(pp_params["num_vars"]):

        # Standard Scale num vars
        pipe_steps.append(
            (
                pp_step_names["STANDARD_SCALER"],
                SklearnTransformerWrapper(
                    StandardScaler(), variables=pp_params["num_vars"]
                ),
            )
        )

    # ===============================================================
    # X column selector
    pipe_steps.append(
        (
            pp_step_names["X_SELECTOR"],
            preprocessors.ColumnSelector(
                columns=pp_params["id_field"], selector_type="drop"
            ),
        )
    )
    # ===============================================================

    pipeline = Pipeline(pipe_steps)

    return pipeline


def save_preprocessor(preprocess_pipe, file_path):
    file_path_and_name = os.path.join(file_path, PREPROCESSOR_FNAME)
    try:
        joblib.dump(preprocess_pipe, file_path_and_name)
    except:
        raise Exception(
            f"""
            Error saving the preprocessor. 
            Does the file path exist {file_path}?"""
        )
    return


def load_preprocessor(file_path):
    file_path_and_name = os.path.join(file_path, PREPROCESSOR_FNAME)
    if not os.path.exists(file_path_and_name):
        raise Exception(
            f"""Error: No trained preprocessor found. 
        Expected to find model files in path: {file_path_and_name}"""
        )

    try:
        preprocess_pipe = joblib.load(file_path_and_name)
    except:
        raise Exception(
            f"""
            Error loading the preprocessor. 
            Do you have the right trained preprocessor at {file_path_and_name}?"""
        )
    return preprocess_pipe
