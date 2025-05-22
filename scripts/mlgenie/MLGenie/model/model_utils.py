###############################################################
## Copyright: PromptBio Corp 2023
# Author: jiayu
# Date of creation: 09/13/2024
# Date of revision:
## AutoML
## Description: The file define the models utils.
#
###############################################################
from typing import List, Dict, Tuple, Union


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score


from .classification import ClsModelType
from .regression import RegressionModelType
from .survival import SurvivalModelType
from ..utils import Metrics, Average, check_X_y, AnalysisType, get_scoring_str


def compare_models(
    train_X: pd.DataFrame,
    train_y: np.ndarray,
    analysis_type: AnalysisType,
    specified_models: List[str] = None,
    metrics: Metrics = Metrics.Accuracy,
    average: Average = Average.Micro,
    random_state: int = 1234,
    cv: int = 5,
    n_jobs: int = 8,
) -> List[Tuple[ClsModelType, float]]:
    """
    Compare the cross-validation performance of all models on the training set.

    Params:
        train_X (DataFrame of shape (n_samlpes, n_features)): Features of train set.
        train_y (numpy matrix of shape (n_samlpes, )): Labels of train set.
        specified_models (List[str]): List of models to be compared.
        analysis_type (AnalysisType): Type of analysis.
        metrics (Metrics = None): Way to measure model's performance.
        random_state (int = None): Pass an int for reproducible results across multiple function calls.
        cv (int = 5): Specify the number of folds in a KFold cross validation.
        n_jobs (int = 8): Number of jobs to run in parallel.
    Returns:
        model_comparison (List[str, float]): All models' cross-validation performance.
    """
    print(f"specified_models: {specified_models}")
    # Check if input are valid.
    check_X_y(train_X, train_y)
    if not isinstance(analysis_type, AnalysisType):
        raise TypeError("analysis_type should be of type AnalysisType!")

    if metrics and not isinstance(metrics, Metrics):
        raise TypeError("metrics should be of type Metrics!")

    if not isinstance(average, Average):
        raise TypeError("average should be of type Average!")

    if not isinstance(cv, int) or cv < 1:
        raise TypeError("cv should be of type int and must be greater then 0!")
    if not isinstance(n_jobs, int) or n_jobs <= 0:
        raise TypeError("n_jobs should be of type int!")

    if specified_models:
        if not isinstance(specified_models, list):
            raise TypeError("specified_models should be of type List!")
        for model in specified_models:
            if (
                analysis_type == AnalysisType.Classification
                and model not in ClsModelType.__members__
            ):
                raise ValueError(f"{model} is not supported!")
            elif (
                analysis_type == AnalysisType.Survival
                and model not in SurvivalModelType.__members__
            ):
                raise ValueError(f"{model} is not supported!")
            elif (
                analysis_type == AnalysisType.Regression
                and model not in RegressionModelType.__members__
            ):
                raise ValueError(f"{model} is not supported!")

    model_comparison = []
    if analysis_type == AnalysisType.Classification:
        total_models = ClsModelType
    elif analysis_type == AnalysisType.Survival:
        total_models = SurvivalModelType
    elif analysis_type == AnalysisType.Regression:
        total_models = RegressionModelType
    else:
        raise ValueError(f"{analysis_type} is not supported!")

    for model_type in total_models:
        if not specified_models or model_type.name in specified_models:
            model = model_type.value
            params = {
                "cv": cv,
                "n_jobs": n_jobs,
                "random_state": random_state,
                "metrics": metrics,
                "average": average,
            }

            valid_params = model.get_params(deep=True)
            fillter_params = {
                key: value for key, value in params.items() if key in valid_params
            }
            model.set_params(**fillter_params)
            try:
                cv_performance = model.k_fold_cross_validation(train_X, train_y)
            except Exception as e:
                cv_performance = 0

            model_comparison.append((model_type, cv_performance))

    return model_comparison
