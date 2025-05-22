#!/usr/bin/env python3
###############################################################
## Copyright: PromptBio Corp 2024
## Author: Wenhao Gu, Jiayu Chen
## Date of creation: 07/24/2024
## Date of revision: 08/06/2024
#
## Project: MLGenie
## Description: This file defines the model class for MLGenie.
##
###############################################################

#!/usr/bin/env python3

import os
from enum import Enum
from typing import List, Dict, Tuple, Union

import pickle
import random
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from ..utils import (
    HPOAlgorithm,
    Metrics,
    Average,
)
from ..hpo_param_config import RegressionParamConfig
from .base import ModelBase


class RegressionModel(ModelBase):
    """
    Class for Regression Base Model
    """

    model_name = None
    model_type = None

    def __init__(
        self,
        metrics: Metrics = Metrics.R2,
        average: Average = Average.Micro,
        hpo_algorithm: HPOAlgorithm = HPOAlgorithm.GridSearch,
        hpo_search_iter: int = 100,
        cv: int = 5,
        random_state: int = 123,
        n_jobs: int = 8,
        path_to_store_model: str = None,
    ):
        super().__init__(
            metrics=metrics,
            average=average,
            hpo_algorithm=hpo_algorithm,
            cv=cv,
            hpo_search_iter=hpo_search_iter,
            random_state=random_state,
            n_jobs=n_jobs,
            path_to_store_model=path_to_store_model,
        )
        self.model = None
        self.param_config = RegressionParamConfig

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ):
        """
        Parameters:
            X (pd.DataFrame): The feature matrix.
            y (pd.Series): The labels of the dataset.

        Returns:
            performance(float): Specific numerical values of model performance.
        """
        performance = self.model.score(X, y)
        return performance

    def predict_proba(self, X):
        """
        Predict the probability of a given data

        Parameters:
            X (pd.DataFrame): The feature matrix.

        Returns:
            predict_probabilities(array-like): The probabilities predicted by model
        """
        raise NotImplementedError("Regression models have no predict_proba method!")


class LassoR(RegressionModel):
    """
    Class for Lasso Regression Model
    """

    model_name = "Lasso Regression"
    model_type = "LassoR"

    def __init__(
        self,
        hpo_algorithm: HPOAlgorithm = HPOAlgorithm.GridSearch,
        hpo_search_iter: int = 100,
        cv: int = 5,
        random_state: int = 123,
        n_jobs: int = 8,
        path_to_store_model: str = None,
    ):
        super().__init__(
            hpo_algorithm=hpo_algorithm,
            cv=cv,
            hpo_search_iter=hpo_search_iter,
            random_state=random_state,
            n_jobs=n_jobs,
            path_to_store_model=path_to_store_model,
        )
        self.model = Lasso(random_state=random_state)


class RidgeR(RegressionModel):
    """
    Class for Ridge Regression model
    """

    model_name = "Ridge Regression"
    model_type = "RidgeR"

    def __init__(
        self,
        hpo_algorithm: HPOAlgorithm = HPOAlgorithm.GridSearch,
        hpo_search_iter: int = 100,
        cv: int = 5,
        random_state: int = 123,
        n_jobs: int = 8,
        path_to_store_model: str = None,
    ):
        super().__init__(
            hpo_algorithm=hpo_algorithm,
            cv=cv,
            hpo_search_iter=hpo_search_iter,
            random_state=random_state,
            n_jobs=n_jobs,
            path_to_store_model=path_to_store_model,
        )
        self.model = Ridge(random_state=random_state)


class ElasticR(RegressionModel):
    """
    Class for Elastic Net Regression model
    """

    model_name = "Elastic Net Regression"
    model_type = "ElasticR"

    def __init__(
        self,
        hpo_algorithm: HPOAlgorithm = HPOAlgorithm.GridSearch,
        hpo_search_iter: int = 100,
        cv: int = 5,
        random_state: int = 123,
        n_jobs: int = 8,
        path_to_store_model: str = None,
    ):
        super().__init__(
            hpo_algorithm=hpo_algorithm,
            cv=cv,
            hpo_search_iter=hpo_search_iter,
            random_state=random_state,
            n_jobs=n_jobs,
            path_to_store_model=path_to_store_model,
        )
        self.model = ElasticNet(random_state=random_state)


class GBDTR(RegressionModel):
    """
    Class for Gradient Boosting Decision Tree Regression model
    """

    model_name = "Gradient Boosting Decision Tree Regression"
    model_type = "GBDTR"

    def __init__(
        self,
        hpo_algorithm: HPOAlgorithm = HPOAlgorithm.GridSearch,
        hpo_search_iter: int = 100,
        cv: int = 5,
        random_state: int = 123,
        n_jobs: int = 8,
        path_to_store_model: str = None,
    ):
        super().__init__(
            hpo_algorithm=hpo_algorithm,
            cv=cv,
            hpo_search_iter=hpo_search_iter,
            random_state=random_state,
            n_jobs=n_jobs,
            path_to_store_model=path_to_store_model,
        )
        self.model = GradientBoostingRegressor(random_state=random_state)


class RFR(RegressionModel):
    """
    Class for Random Forest Regressor model
    """

    model_name = "Random Forest Regressor"
    model_type = "RFR"

    def __init__(
        self,
        hpo_algorithm: HPOAlgorithm = HPOAlgorithm.GridSearch,
        hpo_search_iter: int = 100,
        cv: int = 5,
        random_state: int = 123,
        n_jobs: int = 8,
        path_to_store_model: str = None,
    ):
        super().__init__(
            hpo_algorithm=hpo_algorithm,
            cv=cv,
            hpo_search_iter=hpo_search_iter,
            random_state=random_state,
            n_jobs=n_jobs,
            path_to_store_model=path_to_store_model,
        )
        self.model = RandomForestRegressor(random_state=random_state)


class DTR(RegressionModel):
    """
    Class for Decision Tree Regression model
    """

    model_name = "Decision Tree Regression"
    model_type = "DTR"

    def __init__(
        self,
        hpo_algorithm: HPOAlgorithm = HPOAlgorithm.GridSearch,
        hpo_search_iter: int = 100,
        cv: int = 5,
        random_state: int = 123,
        n_jobs: int = 8,
        path_to_store_model: str = None,
    ):
        super().__init__(
            hpo_algorithm=hpo_algorithm,
            cv=cv,
            hpo_search_iter=hpo_search_iter,
            random_state=random_state,
            n_jobs=n_jobs,
            path_to_store_model=path_to_store_model,
        )
        self.model = DecisionTreeRegressor(random_state=random_state)


class RegressionModelType(Enum):
    """
    Definition of RegressionModelType.
    """

    LassoR = LassoR()
    RidgeR = RidgeR()
    ElasticR = ElasticR()
    GBDTR = GBDTR()
    RFR = RFR()
    DTR = DTR()
