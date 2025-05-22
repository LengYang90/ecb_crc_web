#!/usr/bin/env python3
###############################################################
## Copyright: PromptBio Corp 2024
## Author: Wenhao Gu
## Date of creation: 07/24/2024
#
## Project: MLGenie
## Description: This file defines the feature selection class.
##
###############################################################
from enum import Enum
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    chi2,
    RFE,
    RFECV,
    SelectFromModel,
    f_regression,
)


class FeatureSelectStrategy(Enum):
    """
    Definition of FeatureSelectStrategy.
    """

    Variance = "variance_threshold"
    Chi2 = "chi2"
    RFE = "RFE"
    RFECV = "RFECV"
    Embedded = "embedded"
    Fregression = "f_regression"
    Ranking = "ranking"


class FeatureSelectorBase(ABC):
    def __init__(
        self, n_feat_to_select: int = None, random_state: int = 123, cv: int = 5
    ):
        """
        Initialize the feature selection base.

        Params:
            n_feat_to_select: The number of features to select.
            random_state: The random state.
            cv: The number of cross-validation folds.
        """
        self.n_feat_to_select = n_feat_to_select
        self.random_state = random_state
        self.cv = cv

    def _feature_selection(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        model: BaseEstimator = None,
        strategy: FeatureSelectStrategy = FeatureSelectStrategy.Chi2,
        variance_threshold: float = 0.05,
    ):
        """
        Implements feature selection algorithms including univariate filter selection methods, recursive feature elimination
        and embedded methods.

        Params:
            X (pd.DataFrame): Input features.
            y (np.ndarray): Input labels.
            strategy (FeatureSelectStrategy): strategy to select features.
                if FeatureSelectStrategy.Chi2, then select the n_feat_to_select features with the highest values for the
                    test chi-squared statistic from;
                if FeatureSelectStrategy.RFE, then select features by recursively considering smaller and smaller sets of features;
                if FeatureSelectStrategy.Embedded, then select features according to the random forest's feature_importance.
            n_feat_to_select (int): number of top features to select.
            variance_threshold (float): Features with a training-set variance lower than this threshold will be removed.
            model (BaseEstimator): Specified model for RFE or Embedded method.
            cv (int): Number of folds in RFECV.
        Returns:
            selected_feature(list): selected features.
        """
        # Check the input
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X should be of type pd.DataFrame!")
        if not isinstance(y, np.ndarray):
            raise TypeError("y should be of type np.ndarray!")
        if not isinstance(strategy, FeatureSelectStrategy):
            raise TypeError("strategy should be of type FeatureSelectStrategy!")
        if model is not None and not isinstance(model, BaseEstimator):
            raise TypeError("model should be of type BaseEstimator!")
        if self.n_feat_to_select and not isinstance(self.n_feat_to_select, int):
            raise TypeError("n_feat_to_select should be of type int!")
        if not self.n_feat_to_select and strategy != FeatureSelectStrategy.RFECV:
            n_feat_to_select = len(X.columns)

        if strategy == FeatureSelectStrategy.Ranking:
            model.fit(X, y)
            scores = pd.Series(model.coef_, index=X.columns)
            scores = scores[scores != 0]
            scores = scores.sort_values(ascending=False)
            selected_features = scores[: self.n_feat_to_select].index
        else:
            if strategy == FeatureSelectStrategy.Variance:
                selector = VarianceThreshold(threshold=variance_threshold).fit(X)
            elif strategy == FeatureSelectStrategy.Chi2:
                selector = SelectKBest(chi2, k=self.n_feat_to_select).fit(X, y)
            elif strategy == FeatureSelectStrategy.RFE:
                selector = RFE(
                    estimator=model, n_features_to_select=self.n_feat_to_select
                ).fit(X, y)
            elif strategy == FeatureSelectStrategy.Embedded:
                selector = SelectFromModel(model, max_features=self.n_feat_to_select).fit(
                    X, y
                )
            elif strategy == FeatureSelectStrategy.RFECV:
                selector = RFECV(
                    estimator=model, step=0.001, cv=self.cv, min_features_to_select=1
                ).fit(X, y)
            elif strategy == FeatureSelectStrategy.Fregression:
                selector = SelectKBest(f_regression, k=self.n_feat_to_select).fit(X, y)
            # Get selected features' name
            selected_features = X.columns.values[selector.get_support()]
        return selected_features.tolist()

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def transform(self, X):
        pass

    @abstractmethod
    def fit_transform(self, X, y):
        pass


class ClsFeatureSelector(FeatureSelectorBase):
    def __init__(
        self, n_feat_to_select: int = None, random_state: int = 123, cv: int = 5
    ):
        """
        Initialize the classification feature selector.

        Params:
            n_feat_to_select: The number of features to select.
            random_state: The random state.
            cv: The number of cross-validation folds.
        """
        super().__init__(
            n_feat_to_select=n_feat_to_select, random_state=random_state, cv=cv
        )
        self.selected_features = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """
        Fit the feature selector.

        Params:
            X: The input features.
            y: The input labels.
        """
        candidate_features = []
        if self.n_feat_to_select:
            selectors = {
                "LogisticRegression": LogisticRegression(
                    solver="sag", max_iter=3000, random_state=self.random_state
                ),
                "LinearSVC": LinearSVC(random_state=random_state),
                "RandomForest": RandomForestClassifier(random_state=self.random_state),
                "LDA": LinearDiscriminantAnalysis(),
                "LassoCV": LassoCV(
                    fit_intercept=True,
                    max_iter=1000,
                    random_state=self.random_state,
                    cv=self.cv,
                ),
                # "Lasso1": Lasso(
                #     alpha=1, fit_intercept=True, max_iter=1000, random_state=random_state
                # ),
                # "Lasso.1": Lasso(
                #     alpha=0.1, fit_intercept=True, max_iter=1000, random_state=random_state
                # ),
                "RidgeCV": RidgeCV(fit_intercept=True, cv=self.cv),
            }

            for model_type, model in selectors.items():
                candidate_features.append(
                    (
                        model_type,
                        sorted(
                            self._feature_selection(
                                X,
                                y,
                                strategy=FeatureSelectStrategy.Embedded,
                                model=model,
                            )
                        ),
                    )
                )
        # If n_feat_to_select is not specified, select features by the RFECV.
        else:
            selectors = {
                "LogisticRegression": LogisticRegression(
                    solver="sag", max_iter=3000, random_state=self.random_state
                ),
                "RandomForest": RandomForestClassifier(random_state=self.random_state),
            }
            for model_type, model in selectors.items():
                candidate_features.append(
                    (
                        model_type,
                        sorted(
                            self._feature_selection(
                                X, y, strategy=FeatureSelectStrategy.RFECV, model=model,
                            )
                        ),
                    )
                )

        features_comparison = []
        for selector, features in candidate_features:
            if features:
                # Calculate the mean performance of all models on the selected features.
                model_comparison = compare_models(
                    X.loc[:, features],
                    y,
                    analysis_type=AnalysisType.Classification,
                    metrics=metrics,
                    specified_models=specified_models,
                    cv=self.cv,
                    random_state=self.random_state,
                    n_jobs=n_jobs,
                )
                features_comparison.append((selector, features, model_comparison))
        features_comparison.sort(
            key=lambda t: np.mean([item[1] for item in t[2]]), reverse=True
        )
        selector, selected_features, model_comparison = features_comparison[0]
        self.selected_features = selected_features

        return selected_features

    def transform(self, X: pd.DataFrame):
        """
        Transform the data to select features.

        Params:
            X: The input features.
        Returns:
            X_selected: The selected features.
        """
        return X[self.selected_features]

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray):
        """
        Fit and transform the data to select features.

        Params:
            X: The input features.
            y: The input labels.
        Returns:
            X_selected: The selected features.
        """
        self.fit(X, y)
        return self.transform(X)


class RegFeatureSelector(FeatureSelectorBase):
    def __init__(
        self, n_feat_to_select: int = None, random_state: int = 123, cv: int = 5
    ):
        """
        Initialize the regression feature selector.

        Params:
            n_feat_to_select: The number of features to select.
            random_state: The random state.
            cv: The number of cross-validation folds.
        """
        super().__init__(
            n_feat_to_select=n_feat_to_select, random_state=random_state, cv=cv
        )
        self.selected_features = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """
        Fit the feature selector.

        Params:
            X: The input features.
            y: The input labels.
        """
        candidate_features = []
        if self.n_feat_to_select:
            selectors = {
                "LassoCV": LassoCV(
                    fit_intercept=True,
                    max_iter=1000,
                    random_state=self.random_state,
                    cv=self.cv,
                ),
                "RidgeCV": RidgeCV(fit_intercept=True, cv=self.cv),
                "ElasticNetCV": ElasticNetCV(
                    fit_intercept=True, random_state=self.random_state, cv=self.cv
                ),
            }

            for model_type, model in selectors.items():
                candidate_features.append(
                    (
                        model_type,
                        sorted(
                            self._feature_selection(
                                X,
                                y,
                                strategy=FeatureSelectStrategy.Ranking,
                                model=model,
                                n_feat_to_select=self.n_feat_to_select,
                            )
                        ),
                    )
                )
        # If n_feat_to_select is not specified, select features by the RFECV.
        else:
            selectors = {
                "LinearRegression": LinearRegression(),
                "RandomForest": RandomForestRegressor(random_state=random_state),
            }
            for model_type, model in selectors.items():
                candidate_features.append(
                    (
                        model_type,
                        sorted(
                            self._feature_selection(
                                X, y, strategy=FeatureSelectStrategy.RFECV, model=model,
                            )
                        ),
                    )
                )

        features_comparison = []
        for selector, features in candidate_features:
            # Select top1 model.
            model_comparison = compare_models(
                X.loc[:, features],
                y,
                analysis_type=AnalysisType.Regression,
                metrics=Metrics.R2,
                specified_models=specified_models,
                random_state=self.random_state,
            )
            features_comparison.append((selector, features, model_comparison))
        features_comparison.sort(
            key=lambda t: np.mean([item[1] for item in t[2]]), reverse=True
        )
        selector, selected_features, model_comparison = features_comparison[0]
        self.selected_features = selected_features
        return selected_features

    def transform(self, X: pd.DataFrame):
        """
        Transform the data to select features.

        Params:
            X: The input features.
        Returns:
            X_selected: The selected features.
        """
        return X[self.selected_features]

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray):
        """
        Fit and transform the data to select features.

        Params:
            X: The input features.
            y: The input labels.
        Returns:
            X_selected: The selected features.
        """
        self.fit(X, y)
        return self.transform(X)
