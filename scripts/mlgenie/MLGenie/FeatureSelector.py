#!/usr/bin/env python3
###############################################################
## Copyright: PromptBio Corp 2024
## Author: Minzhe Zhang, Wenhao Gu
## Date of creation: 08/06/2024
## Date of revision: 11/07/2024
#
## Project: MLGenie
## Description: This file defines the class FeatureSelector, which is used to select features.
###############################################################
import copy
import multiprocess as mp
from abc import ABC, abstractmethod
from typing import Union, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.feature_selection import (
    mutual_info_regression,
    mutual_info_classif,
    f_classif,
    f_regression,
    SelectFromModel,
)
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR

from .DataSampler import DataSamplerBootstrap, DataSamplerKFoldCV
from .Utils.Utils import check_X_y


class FeatureSelectorBase(ABC, TransformerMixin):
    """
    Base class for all feature selectors
    """

    def __init__(self, n_features: Union[int, str], random_state: int = 123):
        """
        Initialize the FeatureSelectorBase.

        Params:
            n_features(int or "auto"): Number of features to select.
            random_state(int): Random seed for reproducibility.
        """
        self.n_features = n_features
        self.random_state = random_state
        self.scores_ = None
        self.features_ = None
        self.selected_features_ = None
        self._validate_params()

    def _validate_params(self):
        """
        Validate the parameters for FeatureSelector.
        """
        if (not isinstance(self.n_features, int) and self.n_features != "auto") or (
            isinstance(self.n_features, int) and self.n_features <= 0
        ):
            raise ValueError("`n_features` should be a positive integer or 'auto'.")
        if not isinstance(self.random_state, int):
            raise TypeError("`random_state` should be a integer.")

    @abstractmethod
    def fit(self, X, y):
        """
        Fit the feature selector based on the provided data.
        """
        NotImplementedError("Subclasses should implement this!")

    def transform(self, X) -> pd.DataFrame:
        """
        Reduce DataBase object to the selected features.

        Params:
            X (pd.DataFrame): feature matrix.
        """
        if self.selected_features_ is None:
            raise RuntimeError("Feature selector has not been fitted yet.")
        if X is None or not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if not all([f in X.columns for f in self.selected_features_]):
            raise ValueError("Feature names do not match.")

        return X[self.selected_features_]


class UniveraiteFeatureSelector(FeatureSelectorBase):
    """
    Base class for all univeriate feature selectors
    """

    def __init__(self, n_features: int, random_state: int = 123):
        super().__init__(n_features=n_features, random_state=random_state)

        if self.n_features == "auto":
            raise ValueError(
                "n_features cannot be 'auto' for univariate feature selection."
            )
        self.scorer_ = self._get_univariate_scorer()

    def fit(self, X, y):
        """
        Univariate feature selection based on scoring function

        Params:
            X (pd.DataFrame): feature matrix.
            y (ps.Series): response values.
        Returns:
            self: The fitted feature selector object.
        """
        check_X_y(X, y)

        if self.scorer_ is None:
            raise ValueError("scorer_ must be defined.")

        self.scores_ = self._scoring(X, y)
        # Replace nan with 0
        self.scores_ = np.nan_to_num(self.scores_)
        self.features_ = np.array(X.columns.tolist())
        self.selected_features_ = self._select_top_features()

        return self

    @abstractmethod
    def _scoring(self, X, y):
        """
        The scoring function to calculate score of each features

        Params:
            X (pd.DataFrame): The feature matrix.
            y (pd.Series): The labels of the data.
        """
        NotImplementedError("Subclasses should implement this!")

    @abstractmethod
    def _get_univariate_scorer(self):
        """
        The scoring function to calculate score of each features
        """
        NotImplementedError("Subclasses should implement this!")

    def _select_top_features(self):
        """
        Select the top n_features based on the feature scores

        Returns:
            data (DataBase): The DataBase object containing the feature matrix and metadata.
        """
        if self.scores_ is None:
            raise RuntimeError("Feature scores have not been calculated yet.")

        top_indices = self.scores_.argsort()[-self.n_features :][::-1]
        return self.features_[top_indices]


class MutualInformationSelector(UniveraiteFeatureSelector):
    """
    Feature selector using mutual information for both classification and regression tasks.
    """

    def __init__(self, task_type: str, n_features: int, random_state: int = 123):
        # TODO: Define the task type.
        if task_type not in ["classification", "regression"]:
            raise ValueError(
                "task_type must be either 'classification' or 'regression'"
            )
        self.task_type = task_type
        super().__init__(n_features=n_features, random_state=random_state)

    def _scoring(self, X, y):
        """"
        The scoring function to calculate the mutual information
        """
        return self.scorer_(X, y, random_state=self.random_state)

    def _get_univariate_scorer(self):
        """
        The univerate scoring function to calculate the mutual information
        """
        if self.task_type == "classification":
            return mutual_info_classif
        else:
            return mutual_info_regression


class AUCSelector(UniveraiteFeatureSelector):
    """
    Feature selector using AUC for classification tasks.
    """

    def _get_univariate_scorer(self):
        """
        The univerate scoring function to calculate the AUC
        """

        def auc_scores(X, y):
            if y.unique().shape[0] != 2:
                raise ValueError(
                    "AUC can only be calculated for binary classification."
                )
            return np.abs(
                np.array([roc_auc_score(y, X.iloc[:, i]) for i in range(X.shape[1])])
                - 0.5
            )

        return auc_scores

    def _scoring(self, X, y):
        """"
        The scoring function to calculate the mutual information
        """
        return self.scorer_(X, y)


class FScoreSelector(UniveraiteFeatureSelector):
    """
    Feature selector using ANOVA F score for classification tasks
    """

    def _get_univariate_scorer(self):
        """
        The univerate scoring function to calculate the ANOVA F score
        """
        return f_classif

    def _scoring(self, X, y):
        """"
        The scoring function to calculate the mutual information
        """
        return self.scorer_(X, y)[0]


class CorrelationCoefficientSelector(UniveraiteFeatureSelector):
    """
    Correlation coefficient Selector for regression tasks.
    """

    def _get_univariate_scorer(self):
        """
        The univerate scoring function to calculate the correlation coefficient
        """
        return f_regression

    def _scoring(self, X, y):
        """"
        The scoring function to calculate the mutual information
        """
        # TODO: Check whether abs is needed.
        return np.abs(self.scorer_(X, y)[0])


class ModelBasedSelector(FeatureSelectorBase):
    """
    Base class for all model based feature selectors
    """

    def __init__(
        self, n_features: int, estimator: BaseEstimator, random_state: int = 123,
    ):
        """
        Params:
            # task_type(str): Specify whether the task is 'classification', 'regression' or 'survival'.
            n_features(int): Number of features to select.
            estimator(BaseEstimator): The base estimator for feature selection.
            random_state(int): Random seed for reproducibility.
        """
        super().__init__(n_features=n_features, random_state=random_state)
        if estimator is None or not isinstance(estimator, BaseEstimator):
            raise TypeError("estimator must be a valid sklearn estimator.")

        self.estimator = estimator
        self.model_name = estimator.__class__.__name__
        self.selector_ = self._get_selector()

    @abstractmethod
    def _get_selector(self):
        """
        The model based feature selector
        """
        NotImplementedError("Subclasses should implement this!")

    def fit(self, X, y):
        """
        Fit the feature selector based on the provided data.

        Params:
            X (pd.DataFrame): feature matrix and metadata.
            y (ps.Series): response values.
        Returns:
            self: The fitted feature selector object.
        """
        check_X_y(X, y)

        if self.selector_ is None:
            raise ValueError("selector_ must be defined.")

        self.features_ = np.array(X.columns)

        if isinstance(self.n_features, int) and self.n_features >= len(self.features_):
            self.selected_features_ = self.features_
        else:
            self.selector_.fit(X, y)
            self.selected_features_ = self.features_[
                self.selector_.get_support(indices=True)
            ]

        return self


class EmbeddedSelector(ModelBasedSelector):
    """
    Feature selector using embedded selector.
    """

    def __init__(
        self, n_features: int, estimator: BaseEstimator, random_state: int = 123,
    ):
        super().__init__(
            n_features=n_features, estimator=estimator, random_state=random_state,
        )

    def _get_selector(self):
        """
        Recursive feature eliminator
        """
        return SelectFromModel(
            self.estimator, threshold=None, max_features=self.n_features,
        )


def run_each_bootstrap(bootstrap_data: Tuple):
    """
    Fit the feature selectors on a single bootstrap sample.

    Params:
        bootstrap_data(Tuple): Tuple containing the bootstrap data.
    Returns:
        features_used (List[str]): List of the names of the features used in the bootstrap sample.
        selectors_fitted (List[BaseEstimator]): List of the fitted feature selectors.
    """
    if len(bootstrap_data) != 5:
        raise ValueError("bootstrap_data must be a tuple of length 5.")

    i, X_train, y_train, feature_selectors, random_state = bootstrap_data
    print(f"Fitting bootstrap sample {i + 1} ...", flush=True)

    # Generate a shuffled version of X_train for permutation testing
    if random_state is not None:
        np.random.seed(random_state + i)
    X_shuffle = X_train.apply(np.random.permutation)
    X_shuffle.columns = ["feat_perm" + str(j + 1) for j in range(X_shuffle.shape[1])]

    # Combine original and shuffled features
    X_combine = pd.concat([X_train, X_shuffle], axis=1)

    # Fit each feature selector on the combined data
    selectors_fitted = [
        copy.deepcopy(selector).fit(X_combine, y_train)
        for selector in feature_selectors
    ]

    return X_train.columns.values, selectors_fitted


class FeatureSelector(BaseEstimator):
    """
    Feature selector that integrates various filtering, wrapper, and embedded methods. 
    """

    def __init__(
        self,
        task_type: str,
        n_features: Union[int, str] = "auto",
        random_state: int = 123,
        n_bootstrap: int = 20,
        n_jobs: int = 8,
    ):
        """
        Params:
            task_type(str): Specify whether the task is 'classification', 'regression' or 'survival'.
            n_features(int or str): Number of features to select. If 'auto', the number is determined using forward selection.
            random_state(int, default=123): Random seed for reproducibility.
            n_bootstrap(int, default=20): Number of bootstrap samples to use for feature selection.
            n_jobs(int, default=8): Number of parallel jobs to run.
        """
        super().__init__()
        self.n_bootstrap = n_bootstrap
        self.task_type = task_type
        self.n_features = n_features
        self.random_state = random_state
        self.n_jobs = n_jobs

        # Validate parameters
        self._validate_params()

        # Initialize feature selectors based on problem type
        (
            self.feature_selectors,
            self.selector_weights,
        ) = self._initialize_feature_selectors()
        self.feature_selectors_fitted = []

    def fit(self, X, y):
        """
        Fit the feature selectors to the data using bootstrap sampling and permutation testing.

        Parameters:
        ----------
        X: pd.DataFrame
            The input feature matrix.
        y: pd.Series or np.ndarray
            The target variable.
        """
        # Check the input.
        check_X_y(X, y)

        if X.shape[1] < self.n_features:
            raise ValueError(
                "The number of features to select cannot exceed the number of features in the data."
            )

        # Initialize the data sampler for bootstrap sampling
        self.data = DataSamplerBootstrap(
            X=X,
            y=y,
            n_samples=self.n_bootstrap,
            test_size=0.1,
            feature_size=0.8,
            random_state=self.random_state,
        )

        # Initialize lists to store the features used
        self.features_used = []

        # Prepare the data to pass to each process
        bootstrap_data = [
            (i, X_train, y_train, self.feature_selectors, self.random_state)
            for i, (X_train, X_test, y_train, y_test) in enumerate(self.data)
        ]

        # Use multiprocessing to parallelize the computation.
        with mp.Pool(processes=self.n_jobs) as pool:
            results = pool.map(run_each_bootstrap, bootstrap_data)

        # Collect results
        for features, selectors_fitted in results:
            self.features_used.append(features)
            self.feature_selectors_fitted.append(selectors_fitted)

    def get_selected_features(self):
        """
        Aggregates feature scores from multiple selectors, calculates the 
        false discovery rate (FDR), and returns the top selected features.

        Returns:
            selected_features (List[str]): List of the names of the top selected features.
            selected_scores (np.ndarray): Array of the corresponding feature scores.
        """
        # Get aggregated feature scores and random feature scores
        (
            self.feature_scores_,
            self.random_feature_scores_,
        ) = self._aggregate_features_frequency()

        # Calculate the false discovery rate (FDR) mean and its standard error
        self.feature_scores_fdr_ = np.mean(self.random_feature_scores_)
        self.feature_scores_fdr_std_ = np.std(self.random_feature_scores_) / np.sqrt(
            len(self.feature_selectors)
        )

        # Select the top 'n_features' based on the aggregated scores
        top_features = self.feature_scores_.head(self.n_features)

        return top_features.index.tolist(), top_features.values

    def _validate_params(self):
        """
        Validate the parameters for FeatureSelector.
        """
        if self.task_type not in ["classification", "regression", "survival"]:
            raise ValueError(
                "`task_type` must be 'classification', 'regression' or 'survival'."
            )
        if not isinstance(self.n_features, (int, str)) or (
            isinstance(self.n_features, int) and self.n_features <= 0
        ):
            raise ValueError("`n_features` should be a positive integer or 'auto'.")
        if not isinstance(self.random_state, int) or self.random_state < 0:
            raise ValueError("`random_state` must be a non-negative integer.")
        if not (isinstance(self.n_bootstrap, int) and self.n_bootstrap >= 10):
            raise ValueError("`n_bootstrap` must be a integer larger than 10.")
        if not isinstance(self.n_jobs, int):
            raise TypeError("`n_jobs` must be a integer.")
        if self.n_jobs < 1:
            raise ValueError("`n_jobs` must be a positive integer.")

    def _initialize_feature_selectors(self):
        """
        Initialize feature selectors based on the task type.
        """
        abs_coef = lambda m: np.abs(m.coef_).flatten()

        if self.task_type == "classification":
            base_selectors = {
                "filter": [
                    MutualInformationSelector(
                        task_type=self.task_type, n_features=self.n_features
                    ),
                    AUCSelector(n_features=self.n_features),
                    FScoreSelector(n_features=self.n_features),
                ],
                "embedded": [
                    EmbeddedSelector(
                        n_features=self.n_features,
                        estimator=LogisticRegression(
                            penalty="l1",
                            solver="saga",
                            random_state=self.random_state,
                            max_iter=200,
                        ),
                    ),
                    EmbeddedSelector(
                        n_features=self.n_features,
                        estimator=LinearSVC(
                            dual=True, random_state=self.random_state
                        ),
                    ),
                    EmbeddedSelector(
                        n_features=self.n_features,
                        estimator=RandomForestClassifier(
                            max_depth=10,
                            min_samples_split=5,
                            min_samples_leaf=3,
                            max_leaf_nodes=50,
                            min_impurity_decrease=0.01,
                            ccp_alpha=0.01,
                            bootstrap=True,
                            oob_score=True,
                            random_state=self.random_state,
                        ),
                    ),
                    EmbeddedSelector(
                        n_features=self.n_features,
                        estimator=GradientBoostingClassifier(
                            max_depth=3,
                            min_samples_split=5,
                            min_samples_leaf=3,
                            max_leaf_nodes=50,
                            learning_rate=0.1,
                            n_iter_no_change=5,
                            tol=1e-4,
                            min_impurity_decrease=0.01,
                            subsample=0.9,
                            random_state=self.random_state,
                        ),
                    ),
                ],
            }

        elif self.task_type == "regression":
            base_selectors = {
                "filter": [
                    MutualInformationSelector(
                        task_type=self.task_type, n_features=self.n_features
                    ),
                    CorrelationCoefficientSelector(n_features=self.n_features),
                ],
                "embedded": [
                    EmbeddedSelector(
                        n_features=self.n_features,
                        estimator=Lasso(random_state=self.random_state),
                    ),
                    EmbeddedSelector(
                        n_features=self.n_features,
                        estimator=LinearSVR(random_state=self.random_state),
                    ),
                    EmbeddedSelector(
                        n_features=self.n_features,
                        estimator=RandomForestRegressor(
                            max_depth=10,
                            min_samples_split=5,
                            min_samples_leaf=3,
                            max_leaf_nodes=50,
                            min_impurity_decrease=0.01,
                            ccp_alpha=0.01,
                            bootstrap=True,
                            oob_score=True,
                            random_state=self.random_state,
                        ),
                    ),
                    EmbeddedSelector(
                        n_features=self.n_features,
                        estimator=GradientBoostingRegressor(
                            max_depth=3,
                            min_samples_split=5,
                            min_samples_leaf=3,
                            max_leaf_nodes=50,
                            learning_rate=0.1,
                            n_iter_no_change=5,
                            tol=1e-4,
                            min_impurity_decrease=0.01,
                            subsample=0.9,
                            random_state=self.random_state,
                        ),
                    ),
                ],
            }

        else:
            raise NotImplementedError("Survival estimator not implemented.")

        # Initialize the dictionary of weights based on the selection method
        weights = {"filter": 1, "embedded": 2}

        # Initialize lists to store selectors and their corresponding weights
        selectors = []
        selector_weights = []

        # Iterate through base selectors, assigning and storing the appropriate weight
        for m, s_list in base_selectors.items():
            w = weights[m]
            for s in s_list:
                selectors.append(s)
                selector_weights.append(w)

        # Convert the selector weights to a numpy array
        selector_weights = np.array(selector_weights)

        # Normalize the weights so that they sum to 1
        selector_weights = selector_weights / sum(selector_weights)

        return selectors, selector_weights

    def _aggregate_features_frequency(self):
        """
        Aggregates and ranks features based on their frequency and weights across 
        multiple feature selection iterations. Also estimates the probability of
        selecting features from random permutations for comparison.
        """
        # Initialize a list to store feature selected and associated
        # weights from different selectors in different bootstrap samples
        features_selected = []

        # Iterate through all fitted selectors across different bootstrap samples
        for selectors in self.feature_selectors_fitted:
            features = pd.concat(
                [
                    pd.Series(w, index=s.selected_features_)
                    for s, w in zip(selectors, self.selector_weights)
                ],
                axis=1,
            )

            # Sum the weights for each feature across selectors
            feature_weights = features.fillna(0).sum(axis=1)
            features_selected.append(feature_weights)

        # Combine the feature selection frequency from all folds
        features_selected = pd.concat(features_selected, axis=1)

        # Identify which features were used in bootstrap iterations
        features_used = pd.concat(
            [
                features_selected.index.to_series().isin(f).astype(int)
                for f in self.features_used
            ],
            axis=1,
        )

        # Set weights of used but unselected features to 0
        features_selected[features_selected.isna() & (features_used == 1)] = 0

        # Identify the random shuffled features selected
        idx = features_selected.index.to_series().str.startswith("feat_perm")
        random_features_selected = (
            features_selected.loc[idx, :].fillna(0).values.reshape(-1)
        )

        # Average the frequency across folds and sort features by weight in descending order
        features_selected = (
            features_selected.loc[~idx, :].mean(axis=1).sort_values(ascending=False)
        )

        return features_selected, random_features_selected
