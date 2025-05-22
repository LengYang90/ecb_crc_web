#!/usr/bin/env python3
###############################################################
## Copyright: PromptBio Corp 2024
# Author: whgu
# Date of creation: 07/25/2024
# Date of revision: 09/15/2024
#
## MLGenie
## Description: Unit test for DataProcessor class
#
###############################################################
import os
import sys
import random
import unittest
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.feature_selection import (
    mutual_info_regression,
    mutual_info_classif,
    SequentialFeatureSelector,
    SelectFromModel,
)
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC, LinearSVR

cur_folder = os.path.dirname(os.path.realpath(__file__))
last_folder = os.path.dirname(cur_folder)
sys.path.append(os.path.dirname(last_folder))

from MLGenie.FeatureSelector import (
    FeatureSelectorBase,
    UniveraiteFeatureSelector,
    MutualInformationSelector,
    AUCSelector,
    FScoreSelector,
    CorrelationCoefficientSelector,
    ModelBasedSelector,
    EmbeddedSelector,
    FeatureSelector,
)


class TestFeatureSelectorBase(unittest.TestCase):
    """
    Unit test for FeatureSelectorBase class
    """

    def test_init(self):
        """
        Test the initialization of FeatureSelectorBase class
        """
        with self.assertRaises(TypeError):
            feature_selector = FeatureSelectorBase()


class TestUniveraiteFeatureSelector(unittest.TestCase):
    """
    Unit test for UniveraiteFeatureSelector class
    """

    def test_init(self):
        """
        Test the initialization of UniveraiteFeatureSelector class
        """
        with self.assertRaises(TypeError):
            feature_selector = UniveraiteFeatureSelector()


class TestMutualInformationSelector(unittest.TestCase):
    """
    Unit test for MutualInformationSelector class
    """

    def test_init(self):
        """
        Test the initialization of MutualInformationSelector class
        """
        selector = MutualInformationSelector(task_type="classification", n_features=10)
        self.assertIsInstance(selector, MutualInformationSelector)
        self.assertEqual(selector.task_type, "classification")
        self.assertEqual(selector.n_features, 10)
        self.assertEqual(selector.random_state, 123)

        selector = MutualInformationSelector(
            task_type="regression", n_features=100, random_state=-456
        )
        self.assertIsInstance(selector, MutualInformationSelector)
        self.assertEqual(selector.task_type, "regression")
        self.assertEqual(selector.n_features, 100)
        self.assertEqual(selector.random_state, -456)

        with self.assertRaises(TypeError):
            selector = MutualInformationSelector()
        with self.assertRaises(ValueError):
            selector = MutualInformationSelector(
                task_type="classification", n_features=0
            )
        with self.assertRaises(ValueError):
            selector = MutualInformationSelector(
                task_type="classification", n_features=-1
            )
        with self.assertRaises(ValueError):
            selector = MutualInformationSelector(
                task_type="classification", n_features=10.5
            )
        with self.assertRaises(ValueError):
            selector = MutualInformationSelector(
                task_type="classification", n_features="10"
            )
        with self.assertRaises(ValueError):
            selector = MutualInformationSelector(task_type="test", n_features=10)
        with self.assertRaises(TypeError):
            selector = MutualInformationSelector(
                task_type="classification", n_features=10, random_state="123"
            )
        with self.assertRaises(ValueError):
            selector = MutualInformationSelector(
                task_type="classification", n_features="auto"
            )

    def test_get_univariate_scorer(self):
        """
        Test the get_univariate_scorer method
        """
        selector = MutualInformationSelector(task_type="classification", n_features=10)
        scorer = selector._get_univariate_scorer()
        self.assertEqual(scorer, mutual_info_classif)

        selector = MutualInformationSelector(task_type="regression", n_features=100)
        scorer = selector._get_univariate_scorer()
        self.assertEqual(scorer, mutual_info_regression)

        with self.assertRaises(ValueError):
            selector = MutualInformationSelector(task_type="test", n_features=10)
            scorer = selector._get_univariate_scorer()

    def test_fit(self):
        """
        Test the fit method
        """
        # Test on binary target.
        X = pd.DataFrame(
            {
                "feature1": [0, 1, 0, 1, 0],
                "feature2": [1, 0, 1, 0, 1],
                "feature3": [0, 0, 1, 1, 0],
                "feature4": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature5": [0.1, 10, 0.2, 5, 0.3],
                "feature6": [-10, 0.1, -5, 0.2, -20],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])
        selector = MutualInformationSelector(task_type="classification", n_features=3)
        selector.fit(X, y)
        self.assertEqual(len(selector.selected_features_), 3)
        self.assertEqual(
            set(selector.selected_features_.tolist()),
            {"feature1", "feature2", "feature5"},
        )

        selector = MutualInformationSelector(task_type="classification", n_features=10)
        selector.fit(X, y)
        self.assertEqual(len(selector.selected_features_), 6)

        selector = MutualInformationSelector(task_type="regression", n_features=3)
        selector.fit(X, y)

        # Test with different random state.
        selector = MutualInformationSelector(
            task_type="classification", n_features=3, random_state=456
        )
        selector.fit(X, y)
        self.assertEqual(len(selector.selected_features_), 3)
        self.assertEqual(
            set(selector.selected_features_.tolist()),
            {"feature1", "feature2", "feature6"},
        )

        # Test on continues target.
        X = pd.DataFrame(
            {
                "feature1": [0, 1, 0, 1, 0],
                "feature2": [1, 0, 1, 0, 1],
                "feature3": [0, 0, 1, 1, 0],
                "feature4": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature5": [0.1, 10, 0.2, 5, 0.3],
                "feature6": [-10, 0.1, -5, 0.2, -20],
            }
        )
        y = pd.Series([0.1, 10, 0.2, 5, 0.3])
        selector = MutualInformationSelector(task_type="regression", n_features=3)
        selector.fit(X, y)
        self.assertEqual(len(selector.selected_features_), 3)
        self.assertEqual(
            set(selector.selected_features_.tolist()),
            {"feature2", "feature5", "feature6"},
        )

        selector = MutualInformationSelector(task_type="regression", n_features=10)
        selector.fit(X, y)
        self.assertEqual(len(selector.selected_features_), 6)

        # Test with different random state.
        selector = MutualInformationSelector(
            task_type="regression", n_features=3, random_state=456
        )
        selector.fit(X, y)
        self.assertEqual(len(selector.selected_features_), 3)
        self.assertEqual(
            set(selector.selected_features_.tolist()),
            {"feature2", "feature5", "feature6"},
        )

        selector = MutualInformationSelector(task_type="classification", n_features=10)
        with self.assertRaises(ValueError):
            selector.fit(X, y)

        # Test with nan values.
        X = pd.DataFrame(
            {
                "feature1": [0, 1, 0, 1, 0],
                "feature2": [1, 0, 1, 0, 1],
                "feature3": [0, 0, 1, 1, 0],
                "feature4": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature5": [0.1, 10, 0.2, 5, 0.3],
                "feature6": [-10, 0.1, -5, 0.2, -20],
                "feature7": [0, 1, np.nan, 1, 0],
                "feature8": [np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])
        selector = MutualInformationSelector(task_type="classification", n_features=3)
        with self.assertRaises(ValueError):
            selector.fit(X, y)

        with self.assertRaises(TypeError):
            selector.fit(X)

        # Test invalid input.
        with self.assertRaises(TypeError):
            selector.fit(None, y)
        with self.assertRaises(TypeError):
            selector.fit([1, 2, 3], y)
        with self.assertRaises(TypeError):
            selector.fit(X.values, y)
        with self.assertRaises(TypeError):
            selector.fit(X, None)
        with self.assertRaises(TypeError):
            selector.fit(X, [1, 2, 3])
        with self.assertRaises(TypeError):
            selector.fit(X, y.values)
        with self.assertRaises(ValueError):
            selector.fit(X, y[:-1])

    def test_transform(self):
        """
        Test the transform method
        """
        # Test on binary target.
        X = pd.DataFrame(
            {
                "feature1": [0, 1, 0, 1, 0],
                "feature2": [1, 0, 1, 0, 1],
                "feature3": [0, 0, 1, 1, 0],
                "feature4": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature5": [0.1, 10, 0.2, 5, 0.3],
                "feature6": [-10, 0.1, -5, 0.2, -20],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])
        selector = MutualInformationSelector(task_type="classification", n_features=3)
        with self.assertRaises(RuntimeError):
            selector.transform(X)

        selector.fit(X, y)
        processed_X = selector.transform(X)
        self.assertTrue(processed_X.equals(X[["feature2", "feature1", "feature5"]]))

        X = pd.DataFrame(
            {
                "feature1": [0, 1, 0, 1, 0],
                "feature2": [1, 0, 1, 0, 1],
                "feature5": [1, 0, 1, 0, 1],
            }
        )
        processed_X = selector.transform(X)
        self.assertTrue(processed_X.equals(X[["feature2", "feature1", "feature5"]]))

        with self.assertRaises(TypeError):
            selector.transform(None)
        with self.assertRaises(TypeError):
            selector.transform([1, 2, 3])
        with self.assertRaises(TypeError):
            selector.transform(X.values)
        with self.assertRaises(ValueError):
            selector.transform(X.iloc[:, :-1])

    def test_fit_transform(self):
        """
        Test the fit_transform method
        """
        # Test on binary target.
        X = pd.DataFrame(
            {
                "feature1": [0, 1, 0, 1, 0],
                "feature2": [1, 0, 1, 0, 1],
                "feature3": [0, 0, 1, 1, 0],
                "feature4": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature5": [0.1, 10, 0.2, 5, 0.3],
                "feature6": [-10, 0.1, -5, 0.2, -20],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])
        selector = MutualInformationSelector(task_type="classification", n_features=3)
        processed_X = selector.fit_transform(X, y)
        self.assertTrue(processed_X.equals(X[["feature2", "feature1", "feature5"]]))

        # Try again on the same data.
        processed_X = selector.fit_transform(X, y)
        self.assertTrue(processed_X.equals(X[["feature2", "feature1", "feature5"]]))

        with self.assertRaises(TypeError):
            selector.fit_transform(None, y)
        with self.assertRaises(TypeError):
            selector.fit_transform([1, 2, 3], y)
        with self.assertRaises(TypeError):
            selector.fit_transform(X.values, y)
        with self.assertRaises(TypeError):
            selector.fit_transform(X, None)
        with self.assertRaises(TypeError):
            selector.fit_transform(X, [1, 2, 3])
        with self.assertRaises(TypeError):
            selector.fit_transform(X, y.values)
        with self.assertRaises(ValueError):
            selector.fit_transform(X, y[:-1])


class TestAUCSelector(unittest.TestCase):
    """
    Unit test for AUCSelector class
    """

    def test_init(self):
        selector = AUCSelector(n_features=10)
        self.assertIsInstance(selector, AUCSelector)
        self.assertEqual(selector.n_features, 10)
        self.assertEqual(selector.random_state, 123)

        selector = AUCSelector(n_features=100, random_state=-456)
        self.assertIsInstance(selector, AUCSelector)
        self.assertEqual(selector.n_features, 100)
        self.assertEqual(selector.random_state, -456)

        with self.assertRaises(TypeError):
            selector = AUCSelector()
        with self.assertRaises(ValueError):
            selector = AUCSelector(n_features=0)
        with self.assertRaises(ValueError):
            selector = AUCSelector(n_features=-1)
        with self.assertRaises(ValueError):
            selector = AUCSelector(n_features=10.5)
        with self.assertRaises(ValueError):
            selector = AUCSelector(n_features="10")
        with self.assertRaises(TypeError):
            selector = AUCSelector(n_features=10, random_state="123")
        with self.assertRaises(ValueError):
            selector = AUCSelector(n_features="auto", random_state=123)

    def test_get_univariate_scorer(self):
        """
        Test the get_univariate_scorer method
        """
        selector = AUCSelector(n_features=10)
        scorer = selector._get_univariate_scorer()

    def test_fit(self):
        # Test on binary target.
        X = pd.DataFrame(
            {
                "feature1": [0, 1, 0, 1, 0],
                "feature2": [1, 0, 1, 0, 1],
                "feature3": [0, 0, 1, 1, 0],
                "feature4": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature5": [0.1, 10, 0.2, 5, 0.3],
                "feature6": [-10, 0.1, -5, 0.2, -20],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])
        selector = AUCSelector(n_features=3)
        selector.fit(X, y)
        self.assertEqual(len(selector.selected_features_), 3)
        self.assertEqual(
            set(selector.selected_features_.tolist()),
            {"feature2", "feature5", "feature6"},
        )

        selector = AUCSelector(n_features=10)
        selector.fit(X, y)
        self.assertEqual(len(selector.selected_features_), 6)

        # Test on multi-class target.
        y = pd.Series([0, 1, 2, 1, 0])
        selector = AUCSelector(n_features=3)
        with self.assertRaises(ValueError):
            selector.fit(X, y)

        # Test on continues target.
        y = pd.Series([0.1, 10, 0.2, 5, 0.3])
        selector = AUCSelector(n_features=3)
        with self.assertRaises(ValueError):
            selector.fit(X, y)

        # Test with nan values.
        X = pd.DataFrame(
            {
                "feature1": [0, 1, 0, 1, 0],
                "feature2": [1, 0, 1, 0, 1],
                "feature3": [0, 0, 1, 1, 0],
                "feature4": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature5": [0.1, 10, 0.2, 5, 0.3],
                "feature6": [-10, 0.1, -5, 0.2, -20],
                "feature7": [0, 1, np.nan, 1, 0],
                "feature8": [np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])
        selector = AUCSelector(n_features=3)
        with self.assertRaises(ValueError):
            selector.fit(X, y)

        # Test invalid input.
        with self.assertRaises(TypeError):
            selector.fit(None, y)
        with self.assertRaises(TypeError):
            selector.fit([1, 2, 3], y)
        with self.assertRaises(TypeError):
            selector.fit(X.values, y)
        with self.assertRaises(TypeError):
            selector.fit(X, None)
        with self.assertRaises(TypeError):
            selector.fit(X, [1, 2, 3])
        with self.assertRaises(TypeError):
            selector.fit(X, y.values)
        with self.assertRaises(ValueError):
            selector.fit(X, y[:-1])

    def test_transform(self):
        """
        Test the transform method
        """
        X = pd.DataFrame(
            {
                "feature1": [0, 1, 0, 1, 0],
                "feature2": [1, 0, 1, 0, 1],
                "feature3": [0, 0, 1, 1, 0],
                "feature4": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature5": [0.1, 10, 0.2, 5, 0.3],
                "feature6": [-10, 0.1, -5, 0.2, -20],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])
        selector = AUCSelector(n_features=3)
        with self.assertRaises(RuntimeError):
            selector.transform(X)
        selector.fit(X, y)
        processed_X = selector.transform(X)
        self.assertTrue(processed_X.equals(X[["feature6", "feature5", "feature2"]]))

        X = pd.DataFrame(
            {
                "feature12": [0, 1, 0, 1, 0],
                "feature21": [1, 0, 1, 0, 1],
                "feature53": [1, 0, 1, 0, 1],
            }
        )
        with self.assertRaises(ValueError):
            selector.transform(X)

        with self.assertRaises(TypeError):
            selector.transform(None)
        with self.assertRaises(TypeError):
            selector.transform([1, 2, 3])

    def test_fit_transform(self):
        # Test on binary target.
        X = pd.DataFrame(
            {
                "feature1": [0, 1, 0, 1, 0],
                "feature2": [1, 0, 1, 0, 1],
                "feature3": [0, 0, 1, 1, 0],
                "feature4": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature5": [0.1, 10, 0.2, 5, 0.3],
                "feature6": [-10, 0.1, -5, 0.2, -20],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])
        selector = AUCSelector(n_features=3)
        processed_data = selector.fit_transform(X, y)
        self.assertTrue(processed_data.equals(X[["feature6", "feature5", "feature2"]]))

        # Try again on the same data.
        processed_data = selector.fit_transform(X, y)
        self.assertTrue(processed_data.equals(X[["feature6", "feature5", "feature2"]]))

        # Test on multi-class target.
        y = pd.Series([0, 1, 2, 1, 0])
        selector = AUCSelector(n_features=3)
        with self.assertRaises(ValueError):
            selector.fit_transform(X, y)

        # Test invalid input.
        with self.assertRaises(TypeError):
            selector.fit_transform(None, y)
        with self.assertRaises(TypeError):
            selector.fit_transform([1, 2, 3], y)
        with self.assertRaises(TypeError):
            selector.fit_transform(X.values, y)
        with self.assertRaises(ValueError):
            selector.fit_transform(X, y[:-1])


class TestFScoreSelector(unittest.TestCase):
    """
    Unit test for FScoreSelector class
    """

    def test_init(self):
        selector = FScoreSelector(n_features=10)
        self.assertIsInstance(selector, FScoreSelector)
        self.assertEqual(selector.n_features, 10)
        self.assertEqual(selector.random_state, 123)

        selector = FScoreSelector(n_features=100, random_state=-456)
        self.assertIsInstance(selector, FScoreSelector)
        self.assertEqual(selector.n_features, 100)
        self.assertEqual(selector.random_state, -456)

        with self.assertRaises(TypeError):
            selector = FScoreSelector()
        with self.assertRaises(ValueError):
            selector = FScoreSelector(n_features=0)
        with self.assertRaises(ValueError):
            selector = FScoreSelector(n_features=-1)
        with self.assertRaises(ValueError):
            selector = FScoreSelector(n_features=10.5)
        with self.assertRaises(ValueError):
            selector = FScoreSelector(n_features="10")
        with self.assertRaises(TypeError):
            selector = FScoreSelector(n_features=10, random_state="123")
        with self.assertRaises(ValueError):
            selector = FScoreSelector(n_features="auto", random_state=123)

    def test_get_univariate_scorer(self):
        """
        Test the get_univariate_scorer method
        """
        selector = FScoreSelector(n_features=10)
        scorer = selector._get_univariate_scorer()

    def test_fit(self):
        # Test on binary target.
        X = pd.DataFrame(
            {
                "feature1": [0, 1, 0, 1, 0],
                "feature2": [1, 0, 1, 0, 1],
                "feature3": [0, 0, 1, 1, 0],
                "feature4": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature5": [0.1, 10, 0.2, 5, 0.3],
                "feature6": [-10, 0.1, -5, 0.2, -20],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])
        selector = FScoreSelector(n_features=3)
        selector.fit(X, y)
        self.assertEqual(len(selector.selected_features_), 3)
        self.assertEqual(
            set(selector.selected_features_.tolist()),
            {"feature1", "feature2", "feature5"},
        )

        selector = FScoreSelector(n_features=10)
        selector.fit(X, y)
        self.assertEqual(len(selector.selected_features_), 6)

        # Test on multi-class target.
        y = pd.Series([0, 1, 2, 1, 0])
        selector = FScoreSelector(n_features=3)
        selector.fit(X, y)
        self.assertEqual(len(selector.selected_features_), 3)
        self.assertEqual(
            set(selector.selected_features_.tolist()),
            {"feature1", "feature2", "feature5"},
        )

        # Test on continues target.
        y = pd.Series([0.1, 10, 0.2, 5, 0.3])
        selector = FScoreSelector(n_features=3)
        selector.fit(X, y)
        self.assertEqual(len(selector.selected_features_), 3)

        # Test with nan values.
        X = pd.DataFrame(
            {
                "feature1": [0, 1, 0, 1, 0],
                "feature2": [1, 0, 1, 0, 1],
                "feature3": [0, 0, 1, 1, 0],
                "feature4": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature5": [0.1, 10, 0.2, 5, 0.3],
                "feature6": [-10, 0.1, -5, 0.2, -20],
                "feature7": [0, 1, np.nan, 1, 0],
                "feature8": [np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])
        selector = FScoreSelector(n_features=3)
        with self.assertRaises(ValueError):
            selector.fit(X, y)

        with self.assertRaises(TypeError):
            selector.fit(None, y)
        with self.assertRaises(TypeError):
            selector.fit([1, 2, 3], y)
        with self.assertRaises(TypeError):
            selector.fit(X.values, y)
        with self.assertRaises(TypeError):
            selector.fit(X, None)
        with self.assertRaises(TypeError):
            selector.fit(X, [1, 2, 3])
        with self.assertRaises(TypeError):
            selector.fit(X, y.values)
        with self.assertRaises(ValueError):
            selector.fit(X, y[:-1])

    def test_transform(self):
        """
        Test the transform method
        """
        X = pd.DataFrame(
            {
                "feature1": [0, 1, 0, 1, 0],
                "feature2": [1, 0, 1, 0, 1],
                "feature3": [0, 0, 1, 1, 0],
                "feature4": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature5": [0.1, 10, 0.2, 5, 0.3],
                "feature6": [-10, 0.1, -5, 0.2, -20],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])
        selector = FScoreSelector(n_features=3)
        with self.assertRaises(RuntimeError):
            selector.transform(X)
        selector.fit(X, y)
        processed_data = selector.transform(X)
        self.assertTrue(processed_data.equals(X[["feature2", "feature1", "feature5"]]))

        X = pd.DataFrame(
            {
                "feature12": [0, 1, 0, 1, 0],
                "feature21": [1, 0, 1, 0, 1],
                "feature53": [1, 0, 1, 0, 1],
            }
        )
        with self.assertRaises(ValueError):
            selector.transform(X)

        with self.assertRaises(TypeError):
            selector.transform(None)
        with self.assertRaises(TypeError):
            selector.transform([1, 2, 3])

    def test_fit_transform(self):
        """
        Test the fit_transform method
        """
        # Test on binary target.
        X = pd.DataFrame(
            {
                "feature1": [0, 1, 0, 1, 0],
                "feature2": [1, 0, 1, 0, 1],
                "feature3": [0, 0, 1, 1, 0],
                "feature4": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature5": [0.1, 10, 0.2, 5, 0.3],
                "feature6": [-10, 0.1, -5, 0.2, -20],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])
        selector = FScoreSelector(n_features=3)
        processed_data = selector.fit_transform(X, y)
        self.assertTrue(processed_data.equals(X[["feature2", "feature1", "feature5"]]))

        # Try again with the same data
        processed_data = selector.fit_transform(X, y)
        self.assertTrue(processed_data.equals(X[["feature2", "feature1", "feature5"]]))

        # Test on multi-class target.
        y = pd.Series([0, 1, 2, 1, 0])
        selector = FScoreSelector(n_features=3)
        processed_data = selector.fit_transform(X, y)
        self.assertTrue(processed_data.equals(X[["feature2", "feature1", "feature5"]]))

        # Test on continues target.
        y = pd.Series([0.1, 10, 0.2, 5, 0.3])
        selector = FScoreSelector(n_features=3)
        processed_data = selector.fit_transform(X, y)

        # Test with invalid input.
        with self.assertRaises(TypeError):
            selector.fit_transform(None, y)
        with self.assertRaises(TypeError):
            selector.fit_transform([1, 2, 3], y)
        with self.assertRaises(TypeError):
            selector.fit_transform(X.values, y)
        with self.assertRaises(TypeError):
            selector.fit_transform(X, None)
        with self.assertRaises(TypeError):
            selector.fit_transform(X, [1, 2, 3])
        with self.assertRaises(TypeError):
            selector.fit_transform(X, y.values)
        with self.assertRaises(ValueError):
            selector.fit_transform(X, y[:-1])


class TestCorrelationCoefficientSelector(unittest.TestCase):
    """
    Unit test for CorrelationCoefficientSelector class
    """

    def test_init(self):
        """
        Test the initialization of CorrelationCoefficientSelector class
        """
        selector = CorrelationCoefficientSelector(n_features=10)
        self.assertIsInstance(selector, CorrelationCoefficientSelector)
        self.assertEqual(selector.n_features, 10)
        self.assertEqual(selector.random_state, 123)

        selector = CorrelationCoefficientSelector(n_features=100, random_state=-456)
        self.assertIsInstance(selector, CorrelationCoefficientSelector)
        self.assertEqual(selector.n_features, 100)
        self.assertEqual(selector.random_state, -456)

        with self.assertRaises(TypeError):
            selector = CorrelationCoefficientSelector()
        with self.assertRaises(ValueError):
            selector = CorrelationCoefficientSelector(n_features=0)
        with self.assertRaises(ValueError):
            selector = CorrelationCoefficientSelector(n_features=-1)
        with self.assertRaises(ValueError):
            selector = CorrelationCoefficientSelector(n_features=10.5)
        with self.assertRaises(ValueError):
            selector = CorrelationCoefficientSelector(n_features="10")
        with self.assertRaises(TypeError):
            selector = CorrelationCoefficientSelector(n_features=10, random_state="123")
        with self.assertRaises(ValueError):
            selector = CorrelationCoefficientSelector(
                n_features="auto", random_state=123
            )

    def test_fit(self):
        """
        Test the fit method
        """
        # Test on continues target.
        X = pd.DataFrame(
            {
                "feature1": [0, 1, 0, 1, 0],
                "feature2": [1, 0, 1, 0, 1],
                "feature3": [0, 0, 1, 1, 0],
                "feature4": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature5": [0.1, 10, 0.2, 5, 0.3],
                "feature6": [-10, 0.1, -5, 0.2, -20],
            }
        )
        y = pd.Series([0.1, 10, 0.2, 5, 0.3])
        selector = CorrelationCoefficientSelector(n_features=3)
        selector.fit(X, y)
        self.assertEqual(len(selector.selected_features_), 3)
        self.assertEqual(
            set(selector.selected_features_.tolist()),
            {"feature1", "feature2", "feature5"},
        )

        selector = CorrelationCoefficientSelector(n_features=10)
        selector.fit(X, y)
        self.assertEqual(len(selector.selected_features_), 6)

        # Test on binary target.
        y = pd.Series([0, 1, 0, 1, 0])
        selector = CorrelationCoefficientSelector(n_features=3)
        selector.fit(X, y)
        self.assertEqual(len(selector.selected_features_), 3)
        self.assertEqual(
            set(selector.selected_features_.tolist()),
            {"feature1", "feature2", "feature5"},
        )

        # Test on multi-class target.
        y = pd.Series([0, 1, 2, 1, 0])
        selector = CorrelationCoefficientSelector(n_features=3)
        selector.fit(X, y)
        self.assertEqual(len(selector.selected_features_), 3)

        # Test with nan values.
        X = pd.DataFrame(
            {
                "feature1": [0, 1, 0, 1, 0],
                "feature2": [1, 0, 1, 0, 1],
                "feature3": [0, 0, 1, 1, 0],
                "feature4": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature5": [0.1, 10, 0.2, 5, 0.3],
                "feature6": [-10, 0.1, -5, 0.2, -20],
                "feature7": [0, 1, np.nan, 1, 0],
                "feature8": [np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])
        selector = CorrelationCoefficientSelector(n_features=3)
        with self.assertRaises(ValueError):
            selector.fit(X, y)

        # Test invalid input.
        with self.assertRaises(TypeError):
            selector.fit(None, y)
        with self.assertRaises(TypeError):
            selector.fit([1, 2, 3], y)
        with self.assertRaises(TypeError):
            selector.fit(X.values, y)
        with self.assertRaises(TypeError):
            selector.fit(X, None)
        with self.assertRaises(TypeError):
            selector.fit(X, [1, 2, 3])
        with self.assertRaises(TypeError):
            selector.fit(X, y.values)
        with self.assertRaises(ValueError):
            selector.fit(X, y[:-1])

    def test_transform(self):
        """
        Test the transform method
        """
        X = pd.DataFrame(
            {
                "feature1": [0, 1, 0, 1, 0],
                "feature2": [1, 0, 1, 0, 1],
                "feature3": [0, 0, 1, 1, 0],
                "feature4": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature5": [0.1, 10, 0.2, 5, 0.3],
                "feature6": [-10, 0.1, -5, 0.2, -20],
            }
        )
        y = pd.Series([0.1, 10, 0.2, 5, 0.3])
        selector = CorrelationCoefficientSelector(n_features=3)
        with self.assertRaises(RuntimeError):
            selector.transform(X)
        selector.fit(X, y)
        processed_data = selector.transform(X)
        self.assertTrue(processed_data.equals(X[["feature5", "feature1", "feature2"]]))

        X = pd.DataFrame(
            {
                "feature12": [0, 1, 0, 1, 0],
                "feature21": [1, 0, 1, 0, 1],
                "feature53": [1, 0, 1, 0, 1],
            }
        )
        with self.assertRaises(ValueError):
            selector.transform(X)

        with self.assertRaises(TypeError):
            selector.transform(None)
        with self.assertRaises(TypeError):
            selector.transform([1, 2, 3])
        with self.assertRaises(TypeError):
            selector.transform(X.values)

    def test_fit_transform(self):
        """
        Test the fit_transform method
        """
        # Test on continues target.
        X = pd.DataFrame(
            {
                "feature1": [0, 1, 0, 1, 0],
                "feature2": [1, 0, 1, 0, 1],
                "feature3": [0, 0, 1, 1, 0],
                "feature4": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature5": [0.1, 10, 0.2, 5, 0.3],
                "feature6": [-10, 0.1, -5, 0.2, -20],
            }
        )
        y = pd.Series([0.1, 10, 0.2, 5, 0.3])
        selector = CorrelationCoefficientSelector(n_features=3)
        processed_data = selector.fit_transform(X, y)
        self.assertTrue(processed_data.equals(X[["feature5", "feature1", "feature2"]]))

        # Try again with the same data
        processed_data = selector.fit_transform(X, y)
        self.assertTrue(processed_data.equals(X[["feature5", "feature1", "feature2"]]))

        # Test with invalid input.
        with self.assertRaises(TypeError):
            selector.fit_transform(None, y)
        with self.assertRaises(TypeError):
            selector.fit_transform([1, 2, 3], y)
        with self.assertRaises(TypeError):
            selector.fit_transform(X.values, y)
        with self.assertRaises(TypeError):
            selector.fit_transform(X, None)
        with self.assertRaises(TypeError):
            selector.fit_transform(X, [1, 2, 3])
        with self.assertRaises(TypeError):
            selector.fit_transform(X, y.values)
        with self.assertRaises(ValueError):
            selector.fit_transform(X, y[:-1])


class TestModelBasedSelector(unittest.TestCase):
    """
    Unit test for ModelBasedSelector class
    """

    def test_init(self):
        """
        Test the initialization of ModelBasedSelector class
        """
        with self.assertRaises(TypeError):
            selector = ModelBasedSelector()


# class TestForwardSelector(unittest.TestCase):
#     """
#     Unit test for ForwardSelector class
#     """
#
#     def test_init(self):
#         """
#         Test the initialization of ForwardSelector class
#         """
#         selector = ForwardSelector(
#             task_type="classification",
#             n_features=10,
#             estimator=LogisticRegression(),
#             cv=5,
#         )
#         self.assertIsInstance(selector, ForwardSelector)
#         self.assertEqual(selector.task_type, "classification")
#         self.assertEqual(selector.n_features, 10)
#         self.assertIsInstance(selector.estimator, LogisticRegression)
#         self.assertEqual(selector.cv, 5)
#         self.assertEqual(selector.random_state, 123)
#         self.assertEqual(selector.n_jobs, 8)
#         self.assertIsInstance(selector.selector_, SequentialFeatureSelector)
#         self.assertEqual(selector.selector_.n_features_to_select, 10)
#         self.assertEqual(selector.selector_.cv, 5)
#         self.assertIsInstance(selector.selector_.estimator, LogisticRegression)
#         self.assertEqual(selector.selector_.direction, "forward")
#         self.assertEqual(selector.selector_.scoring, "roc_auc")
#
#         selector = ForwardSelector(
#             task_type="regression",
#             n_features=50,
#             estimator=Lasso(),
#             cv=3,
#             random_state=456,
#             n_jobs=16,
#         )
#         self.assertIsInstance(selector, ForwardSelector)
#         self.assertEqual(selector.task_type, "regression")
#         self.assertEqual(selector.n_features, 50)
#         self.assertIsInstance(selector.estimator, Lasso)
#         self.assertEqual(selector.cv, 3)
#         self.assertEqual(selector.random_state, 456)
#         self.assertEqual(selector.n_jobs, 16)
#         self.assertIsInstance(selector.selector_, SequentialFeatureSelector)
#         self.assertEqual(selector.selector_.n_features_to_select, 50)
#         self.assertEqual(selector.selector_.cv, 3)
#         self.assertIsInstance(selector.selector_.estimator, Lasso)
#         self.assertEqual(selector.selector_.direction, "forward")
#         self.assertEqual(selector.selector_.scoring, "r2")
#
#         selector = ForwardSelector(
#             task_type="classification",
#             n_features="auto",
#             estimator=RandomForestClassifier(),
#             cv=10,
#             random_state=-123,
#         )
#         self.assertIsInstance(selector, ForwardSelector)
#         self.assertEqual(selector.task_type, "classification")
#         self.assertEqual(selector.n_features, "auto")
#         self.assertIsInstance(selector.estimator, RandomForestClassifier)
#         self.assertEqual(selector.cv, 10)
#         self.assertEqual(selector.random_state, -123)
#         self.assertIsInstance(selector.selector_, SequentialFeatureSelector)
#         self.assertEqual(selector.selector_.n_features_to_select, "auto")
#         self.assertEqual(selector.selector_.cv, 10)
#         self.assertIsInstance(selector.selector_.estimator, RandomForestClassifier)
#         self.assertEqual(selector.selector_.direction, "forward")
#         self.assertEqual(selector.selector_.scoring, "roc_auc")
#
#         with self.assertRaises(TypeError):
#             selector = ForwardSelector()
#         with self.assertRaises(ValueError):
#             selector = ForwardSelector(
#                 task_type="test", n_features=10, estimator=LogisticRegression(), cv=5
#             )
#         with self.assertRaises(ValueError):
#             selector = ForwardSelector(
#                 task_type="classification",
#                 n_features=-10,
#                 estimator=LogisticRegression(),
#                 cv=5,
#             )
#         with self.assertRaises(TypeError):
#             selector = ForwardSelector(
#                 task_type="classification", n_features=10, estimator="LR", cv=5
#             )
#         with self.assertRaises(ValueError):
#             selector = ForwardSelector(
#                 task_type="classification",
#                 n_features=10,
#                 estimator=LogisticRegression(),
#                 cv=-5,
#             )
#         with self.assertRaises(ValueError):
#             selector = ForwardSelector(
#                 task_type="classification",
#                 n_features="test",
#                 estimator=LogisticRegression(),
#                 cv=5,
#             )
#         with self.assertRaises(ValueError):
#             selector = ForwardSelector(
#                 task_type="classification",
#                 n_features=10,
#                 estimator=LogisticRegression(),
#                 cv=5,
#                 random_state="123",
#             )
#         with self.assertRaises(ValueError):
#             selector = ForwardSelector(
#                 task_type="classification",
#                 n_features=10,
#                 estimator=LogisticRegression(),
#                 cv=-1,
#                 random_state=123,
#             )
#         with self.assertRaises(ValueError):
#             selector = ForwardSelector(
#                 task_type="classification",
#                 n_features=10,
#                 estimator=LogisticRegression(),
#                 cv=5,
#                 n_jobs="8",
#             )
#         with self.assertRaises(ValueError):
#             selector = ForwardSelector(
#                 task_type="classification",
#                 n_features=10,
#                 estimator=LogisticRegression(),
#                 cv=5,
#                 n_jobs=-2,
#             )
#
#     def test_fit(self):
#         """
#         Test the fit method
#         """
#         # Test on binary target.
#         X = pd.DataFrame(np.random.uniform(0, 1, (100, 10)),)
#         y = pd.Series(np.random.randint(0, 2, 100))
#         data = DataBase(X, y)
#         selector = ForwardSelector(
#             task_type="classification",
#             n_features=5,
#             estimator=LogisticRegression(),
#             cv=5,
#         )
#         with self.assertRaises(ValueError):
#             selector.fit(data)
#         data.update_features(X, layer="processed")
#         selector.fit(data)
#         self.assertEqual(len(selector.selected_features_), 5)
#
#         selector = ForwardSelector(
#             task_type="classification",
#             n_features=10,
#             estimator=LogisticRegression(),
#             cv=5,
#         )
#         selector.fit(data)
#         self.assertEqual(len(selector.selected_features_), 10)
#
#         selector = ForwardSelector(
#             task_type="classification",
#             n_features=15,
#             estimator=LogisticRegression(),
#             cv=5,
#         )
#         selector.fit(data)
#         self.assertEqual(len(selector.selected_features_), 10)
#
#         selector = ForwardSelector(
#             task_type="classification",
#             n_features=5,
#             estimator=RandomForestClassifier(),
#             cv=5,
#         )
#         selector.fit(data)
#         self.assertEqual(len(selector.selected_features_), 5)
#
#         selector = ForwardSelector(
#             task_type="classification",
#             n_features=5,
#             estimator=LogisticRegression(),
#             cv=1,
#         )
#         selector.fit(data)
#         self.assertEqual(len(selector.selected_features_), 5)
#
#         with self.assertRaises(ValueError):
#             selector = ForwardSelector(
#                 task_type="classification",
#                 n_features=10,
#                 estimator=LogisticRegression(),
#                 cv=500,
#             )
#             selector.fit(data)
#
#         selector = ForwardSelector(
#             task_type="classification",
#             n_features=10,
#             estimator=LogisticRegression(),
#             cv=5,
#         )
#         selector.fit(data)
#         self.assertEqual(len(selector.selected_features_), 6)
#
#         #
#
#         # Test on continues target.
#         y = pd.Series([0.1, 10, 0.2, 5, 0.3])
#         data = DataBase(X, y)
#         data.update_features(X, layer="processed")
#         selector = ForwardSelector(
#             task_type="classification",
#             n_features=3,
#             estimator=LogisticRegression(),
#             cv=5,
#         )
#         with self.assertRaises(ValueError):
#             selector.fit(data)


class TestEmbeddedSelector(unittest.TestCase):
    """
    Unit test for EmbeddedSelector class
    """

    def setUp(self):
        np.random.seed(123)

    def test_init(self):
        """
        Test the initialization of EmbeddedSelector class
        """
        selector = EmbeddedSelector(n_features=10, estimator=LogisticRegression())
        self.assertIsInstance(selector, EmbeddedSelector)
        self.assertEqual(selector.n_features, 10)
        self.assertIsInstance(selector.estimator, LogisticRegression)
        self.assertEqual(selector.random_state, 123)
        self.assertIsInstance(selector.selector_, SelectFromModel)

        selector = EmbeddedSelector(
            n_features=100, estimator=RandomForestClassifier(), random_state=456,
        )
        self.assertIsInstance(selector, EmbeddedSelector)
        self.assertEqual(selector.n_features, 100)
        self.assertIsInstance(selector.estimator, RandomForestClassifier)
        self.assertEqual(selector.random_state, 456)
        self.assertIsInstance(selector.selector_, SelectFromModel)

        selector = EmbeddedSelector(n_features=10, estimator=LinearSVC(),)
        self.assertIsInstance(selector, EmbeddedSelector)
        self.assertEqual(selector.n_features, 10)
        self.assertIsInstance(selector.estimator, LinearSVC)
        self.assertEqual(selector.random_state, 123)
        self.assertIsInstance(selector.selector_, SelectFromModel)

        selector = EmbeddedSelector(n_features=10, estimator=RandomForestClassifier(),)
        self.assertIsInstance(selector, EmbeddedSelector)
        self.assertEqual(selector.n_features, 10)
        self.assertIsInstance(selector.estimator, RandomForestClassifier)

        selector = EmbeddedSelector(
            n_features=10, estimator=GradientBoostingClassifier(),
        )
        self.assertIsInstance(selector, EmbeddedSelector)
        self.assertEqual(selector.n_features, 10)
        self.assertIsInstance(selector.estimator, GradientBoostingClassifier)

        selector = EmbeddedSelector(n_features=10, estimator=Lasso(),)

        self.assertIsInstance(selector, EmbeddedSelector)
        self.assertEqual(selector.n_features, 10)
        self.assertIsInstance(selector.estimator, Lasso)

        selector = EmbeddedSelector(n_features=10, estimator=LinearSVR(),)
        self.assertIsInstance(selector, EmbeddedSelector)
        self.assertEqual(selector.n_features, 10)
        self.assertIsInstance(selector.estimator, LinearSVR)

        selector = EmbeddedSelector(n_features=10, estimator=RandomForestRegressor(),)
        self.assertIsInstance(selector, EmbeddedSelector)
        self.assertEqual(selector.n_features, 10)
        self.assertIsInstance(selector.estimator, RandomForestRegressor)

        with self.assertRaises(ValueError):
            selector = EmbeddedSelector(n_features=None, estimator=LogisticRegression())

        with self.assertRaises(ValueError):
            selector = EmbeddedSelector(n_features=-10, estimator=LogisticRegression())

        with self.assertRaises(ValueError):
            selector = EmbeddedSelector(n_features=0, estimator=LogisticRegression())

        with self.assertRaises(ValueError):
            selector = EmbeddedSelector(
                n_features="123", estimator=LogisticRegression()
            )

        with self.assertRaises(TypeError):
            selector = EmbeddedSelector(n_features=10, estimator="LR")

        with self.assertRaises(TypeError):
            selector = EmbeddedSelector(
                n_features=10, estimator=LogisticRegression(), random_state="123"
            )

    def test_fit(self):
        """
        Test the fit method
        """
        # Test on binary target.
        # Test on small dataset.
        X = pd.DataFrame(np.random.uniform(0, 1, (2, 10)))
        y = pd.Series(np.random.randint(0, 2, 2))
        selector = EmbeddedSelector(n_features=2, estimator=LogisticRegression())
        selector.fit(X, y)
        self.assertLessEqual(len(selector.selected_features_), 2)
        self.assertTrue(
            all([feat in X.columns for feat in selector.selected_features_])
        )

        selector = EmbeddedSelector(n_features=4, estimator=RandomForestClassifier())
        selector.fit(X, y)
        self.assertLessEqual(len(selector.selected_features_), 4)
        self.assertTrue(
            all([feat in X.columns for feat in selector.selected_features_])
        )

        selector = EmbeddedSelector(n_features=6, estimator=LinearSVR())
        selector.fit(X, y)
        self.assertLessEqual(len(selector.selected_features_), 6)
        self.assertTrue(
            all([feat in X.columns for feat in selector.selected_features_])
        )

        selector = EmbeddedSelector(
            n_features=8, estimator=GradientBoostingClassifier()
        )
        selector.fit(X, y)
        self.assertLessEqual(len(selector.selected_features_), 8)
        self.assertTrue(
            all([feat in X.columns for feat in selector.selected_features_])
        )
        with self.assertRaises(ValueError):
            selector = EmbeddedSelector(
                n_features=8, estimator=GradientBoostingClassifier(n_iter_no_change=5)
            )
            selector.fit(X, y)

        # Test on medium dataset.
        X = pd.DataFrame(np.random.uniform(0, 1, (100, 100)),)
        y = pd.Series(np.random.randint(0, 2, 100))
        selector = EmbeddedSelector(n_features=5, estimator=LogisticRegression())
        selector.fit(X, y)
        self.assertEqual(len(selector.selected_features_), 5)
        self.assertTrue(
            all([feat in X.columns for feat in selector.selected_features_])
        )

        selector = EmbeddedSelector(n_features=10, estimator=LogisticRegression())
        selector.fit(X, y)
        self.assertEqual(len(selector.selected_features_), 10)
        self.assertTrue(
            all([feat in X.columns for feat in selector.selected_features_])
        )

        selector = EmbeddedSelector(n_features=20, estimator=RandomForestClassifier())
        selector.fit(X, y)
        self.assertEqual(len(selector.selected_features_), 20)
        self.assertTrue(
            all([feat in X.columns for feat in selector.selected_features_])
        )

        selector = EmbeddedSelector(n_features=10, estimator=Lasso())
        selector.fit(X, y)
        self.assertEqual(len(selector.selected_features_), 0)
        self.assertTrue(
            all([feat in X.columns for feat in selector.selected_features_])
        )

        selector = EmbeddedSelector(n_features=10, estimator=LinearSVR())
        selector.fit(X, y)
        self.assertEqual(len(selector.selected_features_), 10)
        self.assertTrue(
            all([feat in X.columns for feat in selector.selected_features_])
        )

        selector = EmbeddedSelector(
            n_features=20, estimator=GradientBoostingClassifier()
        )
        selector.fit(X, y)
        self.assertEqual(len(selector.selected_features_), 20)
        self.assertTrue(
            all([feat in X.columns for feat in selector.selected_features_])
        )

        # Test on continues target.
        # Test on small dataset.
        X = pd.DataFrame(np.random.uniform(0, 1, (2, 10)))
        y = pd.Series(np.random.uniform(0, 10, 2))
        selector = EmbeddedSelector(n_features=2, estimator=LinearSVR())
        selector.fit(X, y)
        self.assertLessEqual(len(selector.selected_features_), 2)
        self.assertTrue(
            all([feat in X.columns for feat in selector.selected_features_])
        )

        selector = EmbeddedSelector(n_features=4, estimator=RandomForestRegressor())
        selector.fit(X, y)
        self.assertLessEqual(len(selector.selected_features_), 4)
        self.assertTrue(
            all([feat in X.columns for feat in selector.selected_features_])
        )

        selector = EmbeddedSelector(n_features=6, estimator=GradientBoostingRegressor())
        selector.fit(X, y)
        self.assertLessEqual(len(selector.selected_features_), 6)
        self.assertTrue(
            all([feat in X.columns for feat in selector.selected_features_])
        )

        selector = EmbeddedSelector(n_features=8, estimator=Lasso())
        selector.fit(X, y)
        self.assertLessEqual(len(selector.selected_features_), 8)
        self.assertTrue(
            all([feat in X.columns for feat in selector.selected_features_])
        )

        # Test on medium dataset.
        X = pd.DataFrame(np.random.uniform(0, 1, (100, 100)),)
        y = pd.Series(np.random.uniform(0, 1, 100))
        selector = EmbeddedSelector(n_features=5, estimator=LinearSVR())
        selector.fit(X, y)
        self.assertEqual(len(selector.selected_features_), 5)
        self.assertTrue(
            all([feat in X.columns for feat in selector.selected_features_])
        )

        selector = EmbeddedSelector(n_features=10, estimator=RandomForestRegressor())
        selector.fit(X, y)
        self.assertEqual(len(selector.selected_features_), 10)
        self.assertTrue(
            all([feat in X.columns for feat in selector.selected_features_])
        )

        selector = EmbeddedSelector(
            n_features=20, estimator=GradientBoostingRegressor()
        )
        selector.fit(X, y)
        self.assertEqual(len(selector.selected_features_), 20)

        selector = EmbeddedSelector(n_features=50, estimator=Lasso())
        selector.fit(X, y)
        self.assertLessEqual(len(selector.selected_features_), 50)

        # Test with nan values.
        X = pd.DataFrame(
            {
                "feature1": [0, 1, 0, 1, 0],
                "feature2": [1, 0, 1, 0, 1],
                "feature3": [0, 0, 1, 1, 0],
                "feature4": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature5": [0.1, 10, 0.2, 5, 0.3],
                "feature6": [-10, 0.1, -5, 0.2, -20],
                "feature7": [0, 1, np.nan, 1, 0],
                "feature8": [np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])
        selector = EmbeddedSelector(n_features=3, estimator=LogisticRegression())
        with self.assertRaises(ValueError):
            selector.fit(X, y)

        with self.assertRaises(TypeError):
            selector.fit(None, y)

        with self.assertRaises(TypeError):
            selector.fit([1, 2, 3], y)

        with self.assertRaises(TypeError):
            selector.fit(X.values, y)

        with self.assertRaises(TypeError):
            selector.fit(X, None)

        with self.assertRaises(TypeError):
            selector.fit(X, [1, 2, 3])

        with self.assertRaises(TypeError):
            selector.fit(X, y.values)

        with self.assertRaises(ValueError):
            selector.fit(X, y[:-1])

        X = pd.DataFrame(np.random.uniform(0, 1, (100, 100)),)
        y = pd.Series(np.random.uniform(0, 10, 100))
        selector = EmbeddedSelector(n_features=20, estimator=RandomForestClassifier())
        with self.assertRaises(ValueError):
            selector.fit(X, y)

        X = pd.DataFrame(np.random.uniform(0, 1, (100, 5)),)
        y = pd.Series(np.random.randint(0, 2, 100))
        selector = EmbeddedSelector(n_features=5, estimator=LogisticRegression())
        selector.fit(X, y)

        self.assertEqual(len(selector.selected_features_), 5)
        selector = EmbeddedSelector(n_features=10, estimator=LogisticRegression())
        selector.fit(X, y)
        self.assertEqual(len(selector.selected_features_), 5)

        # Test on data with too little features.
        X = pd.DataFrame(np.random.uniform(0, 1, (100, 1)),)
        y = pd.Series(np.random.randint(0, 2, 100))
        selector = EmbeddedSelector(n_features=5, estimator=LogisticRegression())
        selector.fit(X, y)
        self.assertEqual(len(selector.selected_features_), 1)
        selector.fit(X.loc[:, []], y)
        self.assertEqual(len(selector.selected_features_), 0)

        # Test reproducibility.
        X = pd.DataFrame(np.random.uniform(0, 1, (100, 100)),)
        y = pd.Series(np.random.randint(0, 2, 100))

        selected_features_n_rounds = []
        for _ in range(5):
            selector = EmbeddedSelector(
                n_features=5,
                estimator=LogisticRegression(
                    random_state=123, penalty="l1", solver="saga", max_iter=200
                ),
            )
            selector.fit(X, y)
            selected_features_n_rounds.append(selector.selected_features_)

        for i in range(1, 5):
            assert np.array_equal(
                np.array(selected_features_n_rounds[0]),
                np.array(selected_features_n_rounds[i]),
            )

        X = pd.DataFrame(np.random.uniform(0, 1, (100, 100)),)
        y = pd.Series(np.random.uniform(0, 10, 100))

        selected_features_n_rounds = []
        for _ in range(5):
            selector = EmbeddedSelector(
                n_features=5, estimator=Lasso(random_state=123),
            )
            selector.fit(X, y)
            selected_features_n_rounds.append(selector.selected_features_)

        for i in range(1, 5):
            assert np.array_equal(
                np.array(selected_features_n_rounds[0]),
                np.array(selected_features_n_rounds[i]),
            )

    def test_transform(self):
        """
        Test the transform method
        """
        X = pd.DataFrame(np.random.uniform(0, 1, (100, 100)),)
        y = pd.Series(np.random.randint(0, 2, 100))
        selector = EmbeddedSelector(n_features=5, estimator=LogisticRegression())

        with self.assertRaises(RuntimeError):
            selector.transform(X)

        selector.fit(X, y)
        processed_X = selector.transform(X)
        self.assertEqual(processed_X.shape[1], 5)
        self.assertTrue(all([feat in X.columns for feat in processed_X.columns]))
        self.assertTrue(
            all([feat in selector.selected_features_ for feat in processed_X.columns])
        )

        with self.assertRaises(TypeError):
            selector.transform(None)
        with self.assertRaises(TypeError):
            selector.transform([1, 2, 3])
        with self.assertRaises(TypeError):
            selector.transform(X.values)

        with self.assertRaises(ValueError):
            selector.transform(
                pd.DataFrame(
                    {"feature1": [0, 1, 0, 1, 0], "feature2": [1, 0, 1, 0, 1],}
                )
            )

    def test_fit_transform(self):
        """
        Test the fit_transform method
        """
        # Test on binary target.
        X = pd.DataFrame(np.random.uniform(0, 1, (100, 100)),)
        y = pd.Series(np.random.randint(0, 2, 100))
        selector = EmbeddedSelector(n_features=5, estimator=LogisticRegression())
        processed_X = selector.fit_transform(X, y)
        self.assertEqual(processed_X.shape[1], 5)
        self.assertTrue(all([feat in X.columns for feat in processed_X.columns]))
        self.assertTrue(
            all([feat in selector.selected_features_ for feat in processed_X.columns])
        )

        selector = EmbeddedSelector(n_features=10, estimator=LinearSVC())
        processed_X = selector.fit_transform(X, y)
        self.assertEqual(processed_X.shape[1], 10)

        selector = EmbeddedSelector(n_features=20, estimator=RandomForestClassifier())
        processed_X = selector.fit_transform(X, y)
        self.assertEqual(processed_X.shape[1], 20)

        selector = EmbeddedSelector(
            n_features=30, estimator=GradientBoostingClassifier()
        )
        processed_X = selector.fit_transform(X, y)
        self.assertEqual(processed_X.shape[1], 26)

        # Test on continues target.
        y = pd.Series(np.random.uniform(0, 10, 100))
        selector = EmbeddedSelector(n_features=5, estimator=Lasso())
        processed_X = selector.fit_transform(X, y)
        self.assertEqual(processed_X.shape[1], 0)

        selector = EmbeddedSelector(n_features=10, estimator=RandomForestRegressor())
        processed_X = selector.fit_transform(X, y)
        self.assertEqual(processed_X.shape[1], 10)

        selector = EmbeddedSelector(n_features=20, estimator=LinearSVR())
        processed_X = selector.fit_transform(X, y)
        self.assertEqual(processed_X.shape[1], 20)

        selector = EmbeddedSelector(n_features=30, estimator=RandomForestRegressor())
        processed_X = selector.fit_transform(X, y)
        self.assertEqual(processed_X.shape[1], 30)

        selector = EmbeddedSelector(
            n_features=40, estimator=GradientBoostingRegressor()
        )
        processed_X = selector.fit_transform(X, y)
        self.assertEqual(processed_X.shape[1], 32)

        X = pd.DataFrame(np.random.uniform(0, 1, (100, 5)),)
        y = pd.Series(np.random.randint(0, 2, 100))
        selector = EmbeddedSelector(n_features=5, estimator=LogisticRegression())
        processed_X = selector.fit_transform(X, y)
        self.assertEqual(processed_X.shape[1], 5)
        self.assertTrue(all([feat in X.columns for feat in processed_X.columns]))
        self.assertTrue(
            all([feat in selector.selected_features_ for feat in processed_X.columns])
        )

        selector = EmbeddedSelector(n_features=10, estimator=LogisticRegression())
        processed_X = selector.fit_transform(X, y)
        self.assertEqual(processed_X.shape[1], 5)
        self.assertTrue(all([feat in X.columns for feat in processed_X.columns]))
        self.assertTrue(
            all([feat in selector.selected_features_ for feat in processed_X.columns])
        )

        with self.assertRaises(TypeError):
            selector.fit_transform(None, y)
        with self.assertRaises(TypeError):
            selector.fit_transform([1, 2, 3], y)
        with self.assertRaises(TypeError):
            selector.fit_transform(X.values, y)
        with self.assertRaises(TypeError):
            selector.fit_transform(X, None)
        with self.assertRaises(TypeError):
            selector.fit_transform(X, [1, 2, 3])
        with self.assertRaises(TypeError):
            selector.fit_transform(X, y.values)
        with self.assertRaises(ValueError):
            selector.fit_transform(X, y[:-1])

        X = pd.DataFrame(np.random.uniform(0, 1, (100, 100)),)
        y = pd.Series(np.random.uniform(0, 10, 100))
        selector = EmbeddedSelector(n_features=10, estimator=RandomForestClassifier())
        with self.assertRaises(ValueError):
            processed_X = selector.fit_transform(X, y)


class TestFeatureSelector(unittest.TestCase):
    """
    Unit test for FeatureSelector class
    """

    def setUp(self):
        random.seed(123)
        os.environ["PYTHONHASHSEED"] = str(123)
        np.random.seed(123)

    def test_init(self):
        """
        Test the initialization of FeatureSelector class
        """
        selector = FeatureSelector(task_type="classification", n_features=10)
        self.assertIsInstance(selector, FeatureSelector)
        self.assertEqual(selector.n_features, 10)
        self.assertEqual(selector.random_state, 123)
        self.assertEqual(selector.n_bootstrap, 20)
        self.assertEqual(selector.task_type, "classification")
        self.assertEqual(len(selector.feature_selectors), 7)
        self.assertEqual(selector.n_jobs, 8)
        assert np.array_equal(
            selector.selector_weights, np.array([1, 1, 1, 2, 2, 2, 2]) / 11
        )
        self.assertEqual(selector.feature_selectors_fitted, [])
        for i, feature_selector in enumerate(selector.feature_selectors):
            if i < 3:
                self.assertIsInstance(feature_selector, UniveraiteFeatureSelector)
            else:
                self.assertIsInstance(feature_selector, EmbeddedSelector)

        selector = FeatureSelector(
            task_type="regression", n_features=20, random_state=456, n_bootstrap=30
        )
        self.assertIsInstance(selector, FeatureSelector)
        self.assertEqual(selector.n_features, 20)
        self.assertEqual(selector.random_state, 456)
        self.assertEqual(selector.n_bootstrap, 30)
        self.assertEqual(selector.task_type, "regression")
        self.assertEqual(len(selector.feature_selectors), 6)
        assert np.array_equal(
            selector.selector_weights, np.array([1, 1, 2, 2, 2, 2]) / 10
        )
        self.assertEqual(selector.feature_selectors_fitted, [])
        for i, feature_selector in enumerate(selector.feature_selectors):
            if i < 2:
                self.assertIsInstance(feature_selector, UniveraiteFeatureSelector)
            else:
                self.assertIsInstance(feature_selector, EmbeddedSelector)

        selector = FeatureSelector(
            task_type="classification",
            n_features=1,
            random_state=123456,
            n_bootstrap=10,
        )
        self.assertIsInstance(selector, FeatureSelector)
        self.assertEqual(selector.n_features, 1)
        self.assertEqual(selector.random_state, 123456)
        self.assertEqual(selector.n_bootstrap, 10)
        self.assertEqual(selector.task_type, "classification")

        selector = FeatureSelector(
            task_type="classification",
            n_features=10000,
            random_state=1,
            n_bootstrap=10000,
        )
        self.assertIsInstance(selector, FeatureSelector)
        self.assertEqual(selector.n_features, 10000)
        self.assertEqual(selector.random_state, 1)
        self.assertEqual(selector.n_bootstrap, 10000)
        self.assertEqual(selector.task_type, "classification")

        selector = FeatureSelector(task_type="classification", n_features=10, n_jobs=1)
        self.assertEqual(selector.n_jobs, 1)

        selector = FeatureSelector(
            task_type="classification", n_features=10, n_jobs=100
        )
        self.assertEqual(selector.n_jobs, 100)

        with self.assertRaises(ValueError):
            selector = FeatureSelector(task_type=None, n_features=10)
        with self.assertRaises(ValueError):
            selector = FeatureSelector(task_type="cls", n_features=10)
        with self.assertRaises(ValueError):
            selector = FeatureSelector(task_type="test", n_features=10)
        with self.assertRaises(ValueError):
            selector = FeatureSelector(task_type=123, n_features=10)
        with self.assertRaises(NotImplementedError):
            selector = FeatureSelector(task_type="survival", n_features=10)

        with self.assertRaises(ValueError):
            selector = FeatureSelector(task_type="classification", n_features=0)
        with self.assertRaises(ValueError):
            selector = FeatureSelector(task_type="classification", n_features=-1)
        with self.assertRaises(ValueError):
            selector = FeatureSelector(task_type="classification", n_features=None)
        with self.assertRaises(ValueError):
            selector = FeatureSelector(task_type="classification", n_features="test")
        with self.assertRaises(ValueError):
            selector = FeatureSelector(task_type="classification", n_features=1.5)

        with self.assertRaises(ValueError):
            selector = FeatureSelector(
                task_type="classification", n_features=10, random_state="123"
            )
        with self.assertRaises(ValueError):
            selector = FeatureSelector(
                task_type="classification", n_features=10, random_state=-1
            )
        with self.assertRaises(ValueError):
            selector = FeatureSelector(
                task_type="classification", n_features=10, random_state=None
            )
        with self.assertRaises(ValueError):
            selector = FeatureSelector(
                task_type="classification", n_features=10, random_state=123.5
            )

        with self.assertRaises(ValueError):
            selector = FeatureSelector(
                task_type="classification", n_features=10, n_bootstrap="10"
            )
        with self.assertRaises(ValueError):
            selector = FeatureSelector(
                task_type="classification", n_features=10, n_bootstrap=-1
            )
        with self.assertRaises(ValueError):
            selector = FeatureSelector(
                task_type="classification", n_features=10, n_bootstrap=None
            )
        with self.assertRaises(ValueError):
            selector = FeatureSelector(
                task_type="classification", n_features=10, n_bootstrap=0
            )
        with self.assertRaises(ValueError):
            selector = FeatureSelector(
                task_type="classification", n_features=10, n_bootstrap=10.5
            )
        with self.assertRaises(ValueError):
            selector = FeatureSelector(
                task_type="classification", n_features=10, n_bootstrap=9
            )
        with self.assertRaises(TypeError):
            selector = FeatureSelector(
                task_type="classification", n_features=10, n_jobs="10"
            )
        with self.assertRaises(ValueError):
            selector = FeatureSelector(
                task_type="classification", n_features=10, n_jobs=0
            )
        with self.assertRaises(ValueError):
            selector = FeatureSelector(
                task_type="classification", n_features=10, n_jobs=-1
            )

    def test_fit(self):
        """
        Test fit() method
        """
        # Test on binary data
        # Test on small data.
        X = pd.DataFrame(
            np.random.uniform(0, 1, (20, 10)),
            columns=[f"feature{i}" for i in range(10)],
        )
        y = pd.Series([1, 0, 1, 0, 1] * 4)
        selector = FeatureSelector(
            task_type="classification", n_features=3, n_bootstrap=10
        )
        selector.fit(X, y)
        self.assertEqual(len(selector.features_used), 10)
        self.assertEqual(len(selector.feature_selectors_fitted), 10)
        for features in selector.features_used:
            self.assertEqual(len(features), 8)
        for feature_selectors in selector.feature_selectors_fitted:
            for feature_selector in feature_selectors:
                self.assertLessEqual(len(feature_selector.selected_features_), 3)

        selector = FeatureSelector(
            task_type="classification", n_features=5, n_bootstrap=20
        )
        selector.fit(X, y)
        self.assertEqual(len(selector.features_used), 20)
        self.assertEqual(len(selector.feature_selectors_fitted), 20)
        for features in selector.features_used:
            self.assertEqual(len(features), 8)
        for feature_selectors in selector.feature_selectors_fitted:
            for feature_selector in feature_selectors:
                self.assertLessEqual(len(feature_selector.selected_features_), 5)

        selector = FeatureSelector(
            task_type="classification", n_features=8, n_bootstrap=50
        )
        selector.fit(X, y)
        self.assertEqual(len(selector.features_used), 50)
        self.assertEqual(len(selector.feature_selectors_fitted), 50)
        for features in selector.features_used:
            self.assertEqual(len(features), 8)
        for feature_selectors in selector.feature_selectors_fitted:
            for feature_selector in feature_selectors:
                self.assertLessEqual(len(feature_selector.selected_features_), 8)

        # Test on medium data.
        X = pd.DataFrame(
            np.random.uniform(0, 1, (100, 1000)),
            columns=[f"feature{i}" for i in range(1000)],
        )
        y = pd.Series(np.random.randint(0, 2, 100))
        selector = FeatureSelector(
            task_type="classification", n_features=5, n_bootstrap=10
        )
        selector.fit(X, y)
        self.assertEqual(len(selector.features_used), 10)
        self.assertEqual(len(selector.feature_selectors_fitted), 10)
        for features in selector.features_used:
            self.assertEqual(len(features), 800)
        for feature_selectors in selector.feature_selectors_fitted:
            for feature_selector in feature_selectors:
                self.assertLessEqual(len(feature_selector.selected_features_), 5)

        selector = FeatureSelector(
            task_type="classification", n_features=10, n_bootstrap=20
        )
        selector.fit(X, y)
        self.assertEqual(len(selector.features_used), 20)
        self.assertEqual(len(selector.feature_selectors_fitted), 20)
        for features in selector.features_used:
            self.assertEqual(len(features), 800)
        for feature_selectors in selector.feature_selectors_fitted:
            for feature_selector in feature_selectors:
                self.assertLessEqual(len(feature_selector.selected_features_), 10)

        selector = FeatureSelector(
            task_type="classification", n_features=50, n_bootstrap=50
        )
        selector.fit(X, y)
        self.assertEqual(len(selector.features_used), 50)
        self.assertEqual(len(selector.feature_selectors_fitted), 50)
        for features in selector.features_used:
            self.assertEqual(len(features), 800)
        for feature_selectors in selector.feature_selectors_fitted:
            for feature_selector in feature_selectors:
                self.assertLessEqual(len(feature_selector.selected_features_), 50)

        # Test on large data.
        X = pd.DataFrame(
            np.random.uniform(0, 1, (1000, 100000)),
            columns=[f"feature{i}" for i in range(100000)],
        )
        y = pd.Series(np.random.randint(0, 2, 1000))
        selector = FeatureSelector(
            task_type="classification", n_features=5, n_bootstrap=10
        )
        selector.fit(X, y)
        self.assertEqual(len(selector.features_used), 10)
        self.assertEqual(len(selector.feature_selectors_fitted), 10)
        for features in selector.features_used:
            self.assertEqual(len(features), 80000)
        for feature_selectors in selector.feature_selectors_fitted:
            for feature_selector in feature_selectors:
                self.assertLessEqual(len(feature_selector.selected_features_), 5)

        # Test on continues data.
        # Test on small data.
        X = pd.DataFrame(
            np.random.uniform(0, 1, (20, 10)),
            columns=[f"feature{i}" for i in range(10)],
        )
        y = pd.Series(np.random.uniform(0, 1, 20))
        selector = FeatureSelector(task_type="regression", n_features=3, n_bootstrap=10)
        selector.fit(X, y)
        self.assertEqual(len(selector.features_used), 10)
        self.assertEqual(len(selector.feature_selectors_fitted), 10)
        for features in selector.features_used:
            self.assertEqual(len(features), 8)
        for feature_selectors in selector.feature_selectors_fitted:
            for feature_selector in feature_selectors:
                self.assertLessEqual(len(feature_selector.selected_features_), 3)

        selector = FeatureSelector(task_type="regression", n_features=5, n_bootstrap=20)
        selector.fit(X, y)
        self.assertEqual(len(selector.features_used), 20)
        self.assertEqual(len(selector.feature_selectors_fitted), 20)
        for features in selector.features_used:
            self.assertEqual(len(features), 8)
        for feature_selectors in selector.feature_selectors_fitted:
            for feature_selector in feature_selectors:
                self.assertLessEqual(len(feature_selector.selected_features_), 5)

        selector = FeatureSelector(task_type="regression", n_features=8, n_bootstrap=50)
        selector.fit(X, y)
        self.assertEqual(len(selector.features_used), 50)
        self.assertEqual(len(selector.feature_selectors_fitted), 50)
        for features in selector.features_used:
            self.assertEqual(len(features), 8)
        for feature_selectors in selector.feature_selectors_fitted:
            for feature_selector in feature_selectors:
                self.assertLessEqual(len(feature_selector.selected_features_), 8)

        # Test on medium data.
        X = pd.DataFrame(
            np.random.uniform(0, 1, (100, 1000)),
            columns=[f"feature{i}" for i in range(1000)],
        )
        y = pd.Series(np.random.uniform(0, 10, 100))
        selector = FeatureSelector(task_type="regression", n_features=5, n_bootstrap=10)
        selector.fit(X, y)
        self.assertEqual(len(selector.features_used), 10)
        self.assertEqual(len(selector.feature_selectors_fitted), 10)
        for features in selector.features_used:
            self.assertEqual(len(features), 800)
        for feature_selectors in selector.feature_selectors_fitted:
            for feature_selector in feature_selectors:
                self.assertLessEqual(len(feature_selector.selected_features_), 5)

        selector = FeatureSelector(
            task_type="regression", n_features=10, n_bootstrap=20
        )
        selector.fit(X, y)
        self.assertEqual(len(selector.features_used), 20)
        self.assertEqual(len(selector.feature_selectors_fitted), 20)
        for features in selector.features_used:
            self.assertEqual(len(features), 800)
        for feature_selectors in selector.feature_selectors_fitted:
            for feature_selector in feature_selectors:
                self.assertLessEqual(len(feature_selector.selected_features_), 10)

        selector = FeatureSelector(
            task_type="regression", n_features=50, n_bootstrap=50
        )
        selector.fit(X, y)
        self.assertEqual(len(selector.features_used), 50)
        self.assertEqual(len(selector.feature_selectors_fitted), 50)
        for features in selector.features_used:
            self.assertEqual(len(features), 800)
        for feature_selectors in selector.feature_selectors_fitted:
            for feature_selector in feature_selectors:
                self.assertLessEqual(len(feature_selector.selected_features_), 50)

        # Test on large data.
        X = pd.DataFrame(
            np.random.uniform(0, 1, (1000, 100000)),
            columns=[f"feature{i}" for i in range(100000)],
        )
        y = pd.Series(np.random.uniform(0, 10, 1000))
        selector = FeatureSelector(task_type="regression", n_features=5, n_bootstrap=10)
        selector.fit(X, y)
        self.assertEqual(len(selector.features_used), 10)
        self.assertEqual(len(selector.feature_selectors_fitted), 10)
        for features in selector.features_used:
            self.assertEqual(len(features), 80000)
        for feature_selectors in selector.feature_selectors_fitted:
            for feature_selector in feature_selectors:
                self.assertLessEqual(len(feature_selector.selected_features_), 5)

        # Test on invalid input.
        X = pd.DataFrame(
            np.random.uniform(0, 1, (10, 10)),
            columns=[f"feature{i}" for i in range(10)],
        )
        y = pd.Series([1, 0, 1, 0, 1] * 4)
        selector = FeatureSelector(
            task_type="classification", n_features=3, n_bootstrap=10
        )
        with self.assertRaises(TypeError):
            selector.fit(None, y)
        with self.assertRaises(TypeError):
            selector.fit("123", y)
        with self.assertRaises(TypeError):
            selector.fit(X.values, y)
        with self.assertRaises(TypeError):
            selector.fit([1, 2, 3], y)
        with self.assertRaises(TypeError):
            selector.fit(X, None)
        with self.assertRaises(TypeError):
            selector.fit(X, "123")
        with self.assertRaises(TypeError):
            selector.fit(X, y.values)
        with self.assertRaises(ValueError):
            selector.fit(X, y[:-1])

        # Test invalid feature name.
        with self.assertRaises(ValueError):
            selector.fit(X, y)

        # Test invalid y
        y = pd.Series([1] * 10)
        with self.assertRaises(ValueError):
            selector.fit(X, y)
        y = pd.Series([0] * 10)
        with self.assertRaises(ValueError):
            selector.fit(X, y)

        y = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
        with self.assertRaises(ValueError):
            selector.fit(X, y)

        y = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
        with self.assertRaises(ValueError):
            selector.fit(X, y)

        # Test on data with too little samples
        selector = FeatureSelector(
            task_type="classification", n_features=3, n_bootstrap=10
        )
        X = pd.DataFrame(
            np.random.uniform(0, 1, (10, 10)),
            columns=[f"feature{i}" for i in range(10)],
        )
        y = pd.Series([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
        with self.assertRaises(ValueError):
            selector.fit(X, y)

        X = pd.DataFrame(np.random.uniform(0, 1, (15, 10)))
        X.columns = [f"feature{i}" for i in range(10)]
        y = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        selector.fit(X, y)

        # Test on data with too little features
        selector = FeatureSelector(
            task_type="classification", n_features=1, n_bootstrap=10
        )
        X = pd.DataFrame(np.random.uniform(0, 1, (15, 1)), columns=["feature0"])
        X.columns = ["feature0"]
        y = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        with self.assertRaises(IndexError):
            selector.fit(X, y)

        X = pd.DataFrame(
            np.random.uniform(0, 1, (15, 2)), columns=["feature0", "feature1"]
        )
        selector.fit(X, y)

        # Test reproducibility.
        X = pd.DataFrame(
            np.random.uniform(0, 1, (100, 100)),
            columns=[f"feature{i}" for i in range(100)],
        )
        y = pd.Series(np.random.randint(0, 2, 100))
        used_features_n_rounds = []
        selected_features_n_rounds = []
        for i in range(5):
            selector = FeatureSelector(
                task_type="classification",
                n_features=5,
                n_bootstrap=10,
                random_state=123,
            )
            selector.fit(X, y)
            used_features_n_rounds.append(selector.features_used)

            selected_features_n_rounds.append(
                [
                    feature_selector.selected_features_
                    for run in selector.feature_selectors_fitted
                    for feature_selector in run
                ]
            )

        for i in range(1, 5):
            assert np.array_equal(
                np.array(used_features_n_rounds[i]), np.array(used_features_n_rounds[0])
            )
            assert np.array_equal(
                np.array(selected_features_n_rounds[i]),
                np.array(selected_features_n_rounds[0]),
            )

        # Test the time with parallel processing.
        X = pd.DataFrame(
            np.random.uniform(0, 1, (100, 1000)),
            columns=[f"feature{i}" for i in range(1000)],
        )
        y = pd.Series(np.random.randint(0, 2, 100))
        selector = FeatureSelector(
            task_type="classification", n_features=5, n_bootstrap=10, n_jobs=2
        )
        start_time = datetime.now()
        selector.fit(X, y)
        end_time = datetime.now()
        time_diff_with_2_jobs = end_time - start_time

        selector = FeatureSelector(
            task_type="classification", n_features=5, n_bootstrap=10, n_jobs=8
        )
        start_time = datetime.now()
        selector.fit(X, y)
        end_time = datetime.now()
        time_diff_with_8_jobs = end_time - start_time
        self.assertGreater(time_diff_with_2_jobs, time_diff_with_8_jobs)

    def test_aggregate_features_frequency(self):
        """
        Test aggregate_features_frequency() method
        """
        # Test on small data.
        # Test with small n_bootstrap.
        X = pd.DataFrame(
            np.random.uniform(0, 1, (20, 10)),
            columns=[f"feature{i}" for i in range(10)],
        )
        y = pd.Series([1, 0, 1, 0, 1] * 4)
        selector = FeatureSelector(
            task_type="classification", n_features=3, n_bootstrap=10
        )
        selector.fit(X, y)
        (
            features_selected,
            random_features_selected,
        ) = selector._aggregate_features_frequency()
        self.assertEqual(features_selected.shape[0], 10)
        self.assertTrue(all([feat in X.columns for feat in features_selected.index]))
        self.assertTrue(all([0 <= value <= 1 for value in features_selected.values]))
        self.assertEqual(random_features_selected.shape[0], 80)
        self.assertTrue(all([0 <= value <= 1 for value in random_features_selected]))
        features_used = []
        for feats in selector.features_used:
            features_used.extend(feats)
        features_used = list(set(features_used))
        self.assertEqual(len(features_used), 10)

        # Test with large n_bootstrap.
        selector = FeatureSelector(
            task_type="classification", n_features=5, n_bootstrap=50
        )
        selector.fit(X, y)
        (
            features_selected,
            random_features_selected,
        ) = selector._aggregate_features_frequency()
        self.assertEqual(features_selected.shape[0], 10)
        self.assertTrue(all([feat in X.columns for feat in features_selected.index]))
        self.assertTrue(all([0 <= value <= 1 for value in features_selected.values]))
        self.assertEqual(random_features_selected.shape[0], 400)
        self.assertTrue(all([0 <= value <= 1 for value in random_features_selected]))
        features_used = []
        for feats in selector.features_used:
            features_used.extend(feats)
        features_used = list(set(features_used))
        self.assertEqual(len(features_used), 10)

        # Test on medium data.
        # Test with small n_bootstrap.
        X = pd.DataFrame(
            np.random.uniform(0, 1, (100, 1000)),
            columns=[f"feature{i}" for i in range(1000)],
        )
        y = pd.Series(np.random.uniform(0, 10, 100))
        selector = FeatureSelector(
            task_type="regression", n_features=10, n_bootstrap=10, random_state=123
        )
        selector.fit(X, y)
        (
            features_selected,
            random_features_selected,
        ) = selector._aggregate_features_frequency()
        self.assertEqual(features_selected.shape[0], 223)
        self.assertTrue(all([feat in X.columns for feat in features_selected.index]))
        self.assertTrue(all([0 <= value <= 1 for value in features_selected.values]))
        self.assertEqual(random_features_selected.shape[0], 600)
        self.assertTrue(all([0 <= value <= 1 for value in random_features_selected]))
        features_used = []
        for feats in selector.features_used:
            features_used.extend(feats)
        features_used = list(set(features_used))
        self.assertEqual(len(features_used), 1000)

        # Test with large n_bootstrap.
        selector = FeatureSelector(
            task_type="regression", n_features=20, n_bootstrap=50, random_state=123
        )
        selector.fit(X, y)
        (
            features_selected,
            random_features_selected,
        ) = selector._aggregate_features_frequency()
        self.assertEqual(features_selected.shape[0], 851)
        self.assertTrue(all([feat in X.columns for feat in features_selected.index]))
        self.assertTrue(all([0 <= value <= 1 for value in features_selected.values]))
        self.assertEqual(random_features_selected.shape[0], 21900)
        self.assertTrue(all([0 <= value <= 1 for value in random_features_selected]))
        features_used = []
        for feats in selector.features_used:
            features_used.extend(feats)
        features_used = list(set(features_used))
        self.assertEqual(len(features_used), 1000)

        # Test on large data.
        X = pd.DataFrame(
            np.random.uniform(0, 1, (1000, 100000)),
            columns=[f"feature{i}" for i in range(100000)],
        )
        y = pd.Series(np.random.randint(0, 2, 1000))
        selector = FeatureSelector(
            task_type="classification", n_features=5, n_bootstrap=10, random_state=123
        )
        selector.fit(X, y)

    def test_get_selected_features(self):
        """
        Test get_selected_features() method
        """
        X = pd.DataFrame(np.random.uniform(0, 1, (20, 10)))
        X.columns = [f"feature{i}" for i in range(10)]
        y = pd.Series([1, 0, 1, 0, 1] * 4)
        selector = FeatureSelector(
            task_type="classification", n_features=3, n_bootstrap=10
        )
        selector.fit(X, y)

        selected_features, weights = selector.get_selected_features()
        self.assertEqual(len(selected_features), 3)
        self.assertTrue(all([feat in X.columns for feat in selected_features]))
        self.assertTrue(all([0 <= value <= 1 for value in weights]))

        # Test reproducibility.
        # Test on binary data.
        X = pd.DataFrame(
            np.random.uniform(0, 1, (100, 100)),
            columns=[f"feature{i}" for i in range(100)],
        )
        y = pd.Series(np.random.randint(0, 2, 100))
        selected_features_n_rounds = []
        for i in range(5):
            selector = FeatureSelector(
                task_type="classification",
                n_features=5,
                n_bootstrap=10,
                random_state=123,
            )
            selector.fit(X, y)
            selected_features, weights = selector.get_selected_features()
            selected_features_n_rounds.append(selected_features)
        for i in range(1, 5):
            self.assertListEqual(
                selected_features_n_rounds[i], selected_features_n_rounds[0]
            )

        # Test on continues data.
        X = pd.DataFrame(
            np.random.uniform(0, 1, (100, 100)),
            columns=[f"feature{i}" for i in range(100)],
        )
        y = pd.Series(np.random.uniform(0, 10, 100))
        selected_features_n_rounds = []
        for i in range(5):
            selector = FeatureSelector(
                task_type="regression", n_features=5, n_bootstrap=10, random_state=123,
            )
            selector.fit(X, y)
            selected_features, weights = selector.get_selected_features()
            selected_features_n_rounds.append(selected_features)
        for i in range(1, 5):
            self.assertListEqual(
                selected_features_n_rounds[i], selected_features_n_rounds[0]
            )

def main() -> None:
    unittest.main()


if __name__ == "__main__":
    main()
