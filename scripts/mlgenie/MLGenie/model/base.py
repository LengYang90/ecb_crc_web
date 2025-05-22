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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from skopt import BayesSearchCV

from ..utils import (
    HPOAlgorithm,
    Metrics,
    Average,
    calculate_prediction_performance,
    get_scoring_str,
    check_X_y,
)


class ModelBase(BaseEstimator, ABC):
    """
    Base model for MLGenie models
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
        """
        Initializes the ModelBase.

        Params:
            metrics (Metrics = Metrics.Accuracy): Way to measure model's performance.
            average (Average = Average.Micro): When the metrics in [Metrics.Fscore, Metrics.Precision,
            hpo_algorithm (HPOAlgorithm = HPOAlgorithm.GridSearch): Approaches to parameter search,
                - If HPOAlgorithm.GridSearch, then exhaustive search over specified parameter values for an estimator;
                - If HPOAlgorithm.RandomSearch, then sample a given number of candidates from a parameter space with a specified distribution;
                - If HPOAlgorithm.BayesianSearch, then apply Bayesian optimization over hyper parameters.
            hpo_search_iter（int = 100) : Number of parameter settings that are sampled when using RandomSearch or BayesianSearch hpo_search_iter trades off runtime vs quality of the solution.
            cv (int = 5): Specify the number of folds in a KFold when search the hyper-parameter space for the best cross validation score.
            random_state (int = 123): Random seed for reproducible results.
            n_job (int = 8): Number of jobs to run in parallel.
            path_to_store_model (str = None): Path to store the model.
        """
        # Check params
        assert (
            isinstance(metrics, Metrics) or metrics is None
        ), "metrics should be of type Metrics!"
        assert isinstance(average, Average), "average should be of type Average!"
        assert isinstance(
            hpo_algorithm, HPOAlgorithm
        ), "hpo_algorithm should be of type HPOAlgorithm!"
        assert (
            isinstance(hpo_search_iter, int) and hpo_search_iter > 0
        ), "hpo_search_iter should be integers that larger than 0 !"
        assert (
            isinstance(cv, int) and cv > 1
        ), "hpo_search_iter should be integers that larger than 1 !"
        assert (
            isinstance(random_state, int) and random_state > 0
        ), "random_state should be integers that larger than 0!"
        assert (
            isinstance(n_jobs, int) and n_jobs > 0
        ), "n_jobs should be integers that larger than 0!"

        # Check the path_to_store_model.
        if path_to_store_model:
            if not isinstance(path_to_store_model, str):
                raise TypeError("path_to_store_model should be of type str!")
            if not os.path.exists(os.path.dirname(path_to_store_model)):
                os.makedirs(os.path.dirname(path_to_store_model))

        self.metrics = metrics
        self.average = average
        self.scoring = get_scoring_str(metrics=metrics, average=average)
        self.hpo_algorithm = hpo_algorithm
        self.hpo_search_iter = hpo_search_iter
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.path_to_store_model = path_to_store_model
        self.model = None
        self.selected_params = None
        self.param_config = None
        self.n_features = None

    def _get_hyper_params_space(self):
        """
        Get the hyper parameter configuration space from the configuration file.

        Returns:
            param_config (List[Dict[str]]): A list of dictionary with parameter names as keys and lists of parameter settings to try as values.
        """
        if self.model_type is None or self.param_config is None:
            raise ValueError("model is not initialized!")
        return self.param_config.get_param_config(
            model_type=self.model_type, hpo_algotithm=self.hpo_algorithm
        )

    def _HPO(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict, float, Dict]:
        """
        Hyperparameter optimization: search the hyper-parameter space for the best cross validation score.

        Params:
            X (pd.DataFrame of shape (n_samples, n_features)): Training vectors.
            y (np.ndarray of shape (n_samples, 1)): Class labels.

        Returns:
            best_params (Dict[str,]): Parameter setting that gave the best results on the hold out data.
            best_score (float): Mean cross-validated score of the best_estimator.
            cv_results (Dict[str,]): Cross-validated score of different parameter settings.
        """
        check_X_y(X, y)
        if self.model is None:
            raise ValueError("Model is not initialized.")

        params_space = self._get_hyper_params_space()

        if self.hpo_algorithm == HPOAlgorithm.GridSearch:
            searchCV = GridSearchCV(
                self.model,
                params_space,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                error_score=0,
            )
        elif self.hpo_algorithm == HPOAlgorithm.RandomSearch:
            searchCV = RandomizedSearchCV(
                self.model,
                params_space,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                n_iter=self.hpo_search_iter,
            )
        elif self.hpo_algorithm == HPOAlgorithm.BayesianSearch:
            searchCV = BayesSearchCV(
                self.model,
                params_space,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                n_iter=self.hpo_search_iter,
            )
        else:
            raise ValueError(
                "Error HPOAlgorithm {}".format(str(self.hpo_algorithm.value))
            )

        searchCV.fit(X, y)
        best_params = searchCV.best_params_
        best_score = searchCV.best_score_
        cv_results = searchCV.cv_results_
        return best_params, best_score, cv_results

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Parameters:
            X (pd.DataFrame): The feature matrix.
            y (pd.Series): The labels of the dataset.

        Returns:
            self: Returns the instance itself.
        """
        check_X_y(X, y)
        if isinstance(y, pd.Series):
            counts = y.value_counts()
            if len(counts) == 2 and self.cv > int(counts.min()):
                raise ValueError(
                    "The cv value {} is greater than sample num {}".format(
                        str(self.cv), str(int(y.value_counts().min()))
                    )
                )

        best_params, best_score, cv_results = self._HPO(X, y)
        self.selected_params = best_params
        self.n_features = len(X.columns)
        self.model.set_params(**best_params)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict the class of a given data

        Parameters:
            X (pd.DataFrame): The feature matrix.

        Returns:
            predict_labels(array-like): The labels predicted by model
        """
        assert isinstance(
            X, pd.DataFrame
        ), "X to be predicted should be of pd.DataFrame type"
        if self.n_features is not None:
            assert (
                len(X.columns) == self.n_features
            ), "columns of X must be equal to {}".format(str(self.n_features))
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict the probability of a given data

        Parameters:
            X (pd.DataFrame): The feature matrix.

        Returns:
            predict_probabilities(array-like): The probabilities predicted by model
        """
        assert isinstance(
            X, pd.DataFrame
        ), "X to be predicted should be of pd.DataFrame type"
        if self.n_features is not None:
            assert (
                len(X.columns) == self.n_features
            ), "columns of X must be equal to {}".format(str(self.n_features))
        return self.model.predict_proba(X)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metrics: Metrics = None,
        average: Average = None,
    ):
        """
        Parameters:
            X (pd.DataFrame): The feature matrix.
            y (pd.Series): The labels of the dataset.

        Returns:
            performance(float): Specific numerical values of model performance.
        """
        check_X_y(X, y)
        if metrics is None:
            metrics = self.metrics
        else:
            assert isinstance(metrics, Metrics), "metrics should be of type Metrics!"
        if average is None:
            average = self.average
        else:
            assert isinstance(average, Average), "average should be of type Average!"
        if metrics == Metrics.AUC:
            y_pred = self.predict_proba(X)
            if set(y) == {0, 1}:
                y_pred = y_pred[:, 1]
            else:
                average = Average.Macro
        else:
            y_pred = self.predict(X)
        performance = calculate_prediction_performance(y, y_pred, metrics, average)
        return performance

    def k_fold_cross_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ):
        """
        This function evaluates model's performance by cross-validation and return
        performance of the model for each run of the cross validation.
        The performance is measured according to the given `metrics` and `average`.

        Detail steps:
        1. The input is split into k=cv smaller sets.
        2. A model is trained using k-1 of the folds as training data.
        3. The resulting model is validated on the remaining part of the data.

        Params:
            X (pd.DataFrame of shape (n_samples, n_features)): Training vectors.
            y (np.ndarray of shape (n_samples, 1)): Class labels.
            cv (int = 5): Specify the number of folds in a KFold.
            n_job (int = 8): Number of jobs to run in parallel.

        Return:
            k_fold_performance (np.ndarray of float of shape=cv): Array of performance of the model for
                each run of the cross validation.
        """
        check_X_y(X, y)
        scoring = get_scoring_str(self.metrics, self.average)
        k_fold_performance = cross_val_score(
            self.model,
            X,
            y,
            scoring=scoring,
            cv=self.cv,
            n_jobs=self.n_jobs,
            error_score=0,
        )
        return k_fold_performance.mean()

    def save(self, path: str = None):
        """
        Save the model file.

        Parameters:
            path (str): The path to store the model file. If None, use self.path_to_store_model.

        """

        if path:
            if not isinstance(path, str):
                raise ValueError("{} should be of type string!".format(str(path)))
            if not os.path.exists(os.path.dirname(path)):
                raise ValueError("{} is not exists!".format(str(path)))
            save_path = path
        else:
            if self.path_to_store_model:
                save_path = self.path_to_store_model
            else:
                raise ValueError("path_to_store_model is None, failed to save model！")

        with open(save_path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str):
        """
        Load model from given model file

        Parameters:
            path (str): The path to store the model file.
        """
        if not isinstance(path, str):
            raise ValueError("{} should be of type string!".format(str(path)))
        if not os.path.exists(path):
            raise ValueError("{} is not exists!".format(str(path)))

        with open(path, "rb") as f:
            self.model = pickle.load(f)
