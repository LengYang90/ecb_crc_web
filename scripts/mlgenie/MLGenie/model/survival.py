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
from sksurv.svm import FastSurvivalSVM
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import (
    RandomSurvivalForest,
    GradientBoostingSurvivalAnalysis,
)

from ..utils import HPOAlgorithm, Metrics, Average, convert_survival_label
from ..hpo_param_config import SurvivalParamConfig
from .base import ModelBase


class SurvivalModel(ModelBase):
    """
    Cl ass for Survival Base Model
    """

    model_name = None
    model_type = None

    def __init__(
        self,
        metrics: Metrics = None,
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
        self.param_config = SurvivalParamConfig

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Parameters:
            X (pd.DataFrame): The feature matrix.
            y (pd.Series): The labels of the dataset.

        Returns:
            self: Returns the instance itself.
        """
        y = convert_survival_label(y)
        return super().fit(X, y)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ):
        """
        Parameters:
            X (pd.DataFrame): The feature matrix.
            y (pd.DataFrame): The labels of the dataset.

        Returns:
            performance(float): Specific numerical values of model performance.
        """
        y = convert_survival_label(y)
        performance = self.model.score(X, y)
        return performance


class CoxPH(SurvivalModel):
    """
    Class for Cox Proportional Hazard Model
    """

    model_name = "Cox Proportional Hazard"
    model_type = "CoxPH"

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
        self.model = CoxPHSurvivalAnalysis()


class Coxnet(SurvivalModel):
    """
    Class for Coxnet Model
    """

    model_name = "Coxnet"
    model_type = "Coxnet"

    def __init__(
        self,
        hpo_algorithm: HPOAlgorithm = HPOAlgorithm.GridSearch,
        hpo_search_iter: int = 100,
        cv: int = 5,
        random_state: int = 123,
        n_jobs: int = 8,
        path_to_store_model: str = None,
        l1_ratio: float = 1.0,
        alpha_min_ratio: float = 0.01,
    ):
        super().__init__(
            hpo_algorithm=hpo_algorithm,
            cv=cv,
            hpo_search_iter=hpo_search_iter,
            random_state=random_state,
            n_jobs=n_jobs,
            path_to_store_model=path_to_store_model,
        )
        self.l1_ratio = l1_ratio
        self.alpha_min_ratio = alpha_min_ratio
        self.model = CoxnetSurvivalAnalysis(
            l1_ratio=self.l1_ratio, alpha_min_ratio=self.alpha_min_ratio
        )


class RSF(SurvivalModel):
    """
    Class for Random Survival Forest Model
    """

    model_name = "Random Survival Forest"
    model_type = "RSF"

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
        self.model = RandomSurvivalForest(random_state=random_state)


class GBS(SurvivalModel):
    """
    Class for Gradient Boosting Survival Analysis Model
    """

    model_name = "Gradient Boosting Survival Analysis"
    model_type = "GBS"

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
        self.model = GradientBoostingSurvivalAnalysis(random_state=random_state)


class FSVM(SurvivalModel):
    """
    Class for Gradient Boosting Survival Analysis Model
    """

    model_name = "Fast Survival Support Vector Machine"
    model_type = "FSVM"

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
        self.model = FastSurvivalSVM(random_state=random_state)


class SurvivalModelType(Enum):
    """
    Definition of SurvivalModelType.
    """

    CoxPH = CoxPH()
    Coxnet = Coxnet()
    RSF = RSF()
    GBS = GBS()
    FSVM = FSVM()
