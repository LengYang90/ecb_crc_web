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
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from ..utils import (
    HPOAlgorithm,
    Metrics,
    Average,
)
from ..hpo_param_config import ClsParamConfig
from .base import ModelBase


class ClsModel(ModelBase):
    """
    Class for Classification Base Model
    """

    model_name = None
    model_type = None

    def __init__(
        self,
        metrics: Metrics = Metrics.Accuracy,
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
        self.param_config = ClsParamConfig


class SVM(ClsModel):
    """
    Class for Support Vector Machine
    """

    model_name = "Support Vector Machine"
    model_type = "SVM"

    def __init__(
        self,
        metrics: Metrics = Metrics.Accuracy,
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

        self.model = SVC(random_state=random_state, probability=True)


class LR(ClsModel):
    """
    Class for Logistic Regression
    """

    model_name = "LogisticRegression"
    model_type = "LR"

    def __init__(
        self,
        metrics: Metrics = Metrics.Accuracy,
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

        self.model = LogisticRegression(random_state=random_state)


class RF(ClsModel):
    """
    Class for Random Forest Classifier
    """

    model_name = "Random Forest Classifier"
    model_type = "RF"

    def __init__(
        self,
        metrics: Metrics = Metrics.Accuracy,
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

        self.model = RandomForestClassifier(random_state=random_state)


class KNN(ClsModel):
    """
    Class for KNeighbors Classifier
    """

    model_name = "KNeighbors Classifier"
    model_type = "KNN"

    def __init__(
        self,
        metrics: Metrics = Metrics.Accuracy,
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

        self.model = KNeighborsClassifier()


class DT(ClsModel):
    """
    Class for Decision Tree Classifier
    """

    model_name = "Decision Tree Classifier"
    model_type = "DT"

    def __init__(
        self,
        metrics: Metrics = Metrics.Accuracy,
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

        self.model = DecisionTreeClassifier(random_state=random_state)


class GBDT(ClsModel):
    """
    Class for Gradient Boosting Classifier
    """

    model_name = "Gradient Boosting Classifier"
    model_type = "GBDT"

    def __init__(
        self,
        metrics: Metrics = Metrics.Accuracy,
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

        self.model = GradientBoostingClassifier(random_state=random_state)


class MLP(ClsModel):
    """
    Class for MLPClassifier
    """

    model_name = "MLPClassifier"
    model_type = "MLP"

    def __init__(
        self,
        metrics: Metrics = Metrics.Accuracy,
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

        self.model = MLPClassifier(random_state=random_state)


class ClsModelType(Enum):
    """
    Definition of ModelType of classification.
    """

    SVM = SVM()
    LR = LR()
    RF = RF()
    KNN = KNN()
    DT = DT()
    GBDT = GBDT()
    MLP = MLP()
