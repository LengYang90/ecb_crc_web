#!/usr/bin/env python3
###############################################################
## Copyright: PromptBio Corp 2024
# Author: whgu
# Date of creation: 07/25/2024
# Date of revision: 11/14/2024
#
## MLGenie
## Description: Unit test for DataProcessor class
#
###############################################################
import os
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


cur_folder = os.path.dirname(os.path.realpath(__file__))
last_folder = os.path.dirname(cur_folder)
sys.path.append(os.path.dirname(last_folder))

from MLGenie.Data import DataBase
from MLGenie.DataProcessor import (
    DataProcessorBase,
    BinaryDataProcessor,
    ContinuesDataProcessor,
    HighQualityFeatureFilter,
    InformativeFeatureFilter,
    MissingValueImputer,
    SkewnessCorrector,
    OutlierCorrector,
    FeatureScaler,
    CorrelatedFeatureFilter,
)


class TestHighQualityFeatureFilter(unittest.TestCase):
    """
    Test HighQualityFeatureFilter class.
    """

    def setUp(self) -> None:
        """
        """
        pass

    def test_init(self):
        """
        Test __init__ method.
        """
        # Test with default values.
        filter = HighQualityFeatureFilter()
        self.assertIsInstance(filter, HighQualityFeatureFilter)
        self.assertIsNone(filter.null_ratio_threshold)
        self.assertIsNone(filter.dominance_threshold)
        self.assertIsNone(filter.variance_threshold)
        self.assertIsNone(filter.selected_features_)

        # Test with specified values.
        filter = HighQualityFeatureFilter(
            null_ratio_threshold=0.1, dominance_threshold=0.9, variance_threshold=0.3
        )
        self.assertIsInstance(filter, HighQualityFeatureFilter)
        self.assertEqual(filter.null_ratio_threshold, 0.1)
        self.assertEqual(filter.dominance_threshold, 0.9)
        self.assertEqual(filter.variance_threshold, 0.3)
        self.assertIsNone(filter.selected_features_)

        # Test with invalid values.
        with self.assertRaises(TypeError):
            filter = HighQualityFeatureFilter(null_ratio_threshold="1")
        with self.assertRaises(TypeError):
            filter = HighQualityFeatureFilter(dominance_threshold="0.5")
        with self.assertRaises(TypeError):
            filter = HighQualityFeatureFilter(variance_threshold="-1")

        with self.assertRaises(ValueError):
            filter = HighQualityFeatureFilter(null_ratio_threshold=-1)
        with self.assertRaises(ValueError):
            filter = HighQualityFeatureFilter(dominance_threshold=-1)
        with self.assertRaises(ValueError):
            filter = HighQualityFeatureFilter(variance_threshold=-1)

        with self.assertRaises(ValueError):
            filter = HighQualityFeatureFilter(null_ratio_threshold=1)
        with self.assertRaises(ValueError):
            filter = HighQualityFeatureFilter(null_ratio_threshold=2)
        with self.assertRaises(ValueError):
            filter = HighQualityFeatureFilter(dominance_threshold=0)
        with self.assertRaises(ValueError):
            filter = HighQualityFeatureFilter(dominance_threshold=0.4)
        with self.assertRaises(ValueError):
            filter = HighQualityFeatureFilter(dominance_threshold=2)

        # Test with valid values.
        filter = HighQualityFeatureFilter(null_ratio_threshold=0)
        self.assertEqual(filter.null_ratio_threshold, 0)

        filter = HighQualityFeatureFilter(dominance_threshold=0.5)
        self.assertEqual(filter.dominance_threshold, 0.5)
        filter = HighQualityFeatureFilter(dominance_threshold=1)
        self.assertEqual(filter.dominance_threshold, 1)

        filter = HighQualityFeatureFilter(variance_threshold=0)
        self.assertEqual(filter.variance_threshold, 0)
        filter = HighQualityFeatureFilter(variance_threshold=100)
        self.assertEqual(filter.variance_threshold, 100)

    def test_identify_high_quality_features(self):
        """
        Test identify_high_quality_features method.
        """
        # Test with default values.
        filter = HighQualityFeatureFilter()

        # Test with invalid values.
        with self.assertRaises(TypeError):
            filter._identify_high_quality_features(None)
        with self.assertRaises(TypeError):
            filter._identify_high_quality_features("test")
        with self.assertRaises(TypeError):
            filter._identify_high_quality_features(1)

        # Test with valid values.
        X = pd.DataFrame(
            {
                "feature1": [1, 1, 1],
                "feature2": [np.nan, np.nan, np.nan],
                "feature3": [7, 8, 9],
            }
        )
        selected_features = filter._identify_high_quality_features(X)

        self.assertIsInstance(selected_features, pd.Index)
        self.assertListEqual(
            selected_features.tolist(), ["feature1", "feature2", "feature3"]
        )

        # Test with specified values.
        filter = HighQualityFeatureFilter(
            null_ratio_threshold=0.5, dominance_threshold=0.5, variance_threshold=0.1
        )
        X = pd.DataFrame(
            {
                "feature1": [1, np.nan, np.nan, 2],
                "feature2": [1, np.nan, np.nan, np.nan],
                "feature3": [1, 1, 0, 0],
                "feature4": [1, 1, 1, 0],
                "feature5": [1, 1.001, 1.002, 1.003],
                "feature6": [1.5, 2.5, 3.6, 4.8],
            }
        )
        selected_features = filter._identify_high_quality_features(X)
        self.assertListEqual(
            selected_features.tolist(), ["feature1", "feature3", "feature6"]
        )

        filter = HighQualityFeatureFilter(null_ratio_threshold=0)
        X = pd.DataFrame({"feature1": [1, np.nan, 2, 3], "feature2": [1, 1, 1, 1],})
        selected_features = filter._identify_high_quality_features(X)
        self.assertListEqual(selected_features.tolist(), ["feature2"])

        filter = HighQualityFeatureFilter(dominance_threshold=0.95)
        X = pd.DataFrame(
            {
                "feature1": [1, np.nan, 2, 3, 5, 6, 7, 8, 9, 10],
                "feature2": [1, 1, 1, 1, 1, 1, 1, 1, 1, np.nan],
                "feature3": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                "feature4": [1, 1, 1, 1, 1, 1, 1, 1, 0, np.nan],
                "feature5": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            }
        )
        selected_features = filter._identify_high_quality_features(X)
        self.assertListEqual(
            selected_features.tolist(), ["feature1", "feature4", "feature5"]
        )

        filter = HighQualityFeatureFilter(variance_threshold=10)
        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4],
                "feature2": [1, 1000, 10000, np.nan],
                "feature3": [1, 1000, 10000, 100000],
            }
        )
        selected_features = filter._identify_high_quality_features(X)
        self.assertListEqual(selected_features.tolist(), ["feature2", "feature3"])

        filter = HighQualityFeatureFilter(
            null_ratio_threshold=0.1, dominance_threshold=0.5, variance_threshold=0.1
        )
        X = pd.DataFrame(
            {
                "feature1": [1, np.nan, np.nan, 2],
                "feature2": [1, np.nan, np.nan, np.nan],
                "feature4": [1, 1, 1, 0],
                "feature5": [1, 1.001, 1.002, 1.003],
            }
        )
        selected_features = filter._identify_high_quality_features(X)
        self.assertListEqual(selected_features.tolist(), [])

    def test_fit(self):
        """
        Test fit method.
        """
        feature_matrix = pd.DataFrame(
            {
                "feature1": [1, np.nan, np.nan, 2],
                "feature2": [1, np.nan, np.nan, np.nan],
                "feature3": [1, 1, 0, 0],
                "feature4": [1, 1, 1, 0],
                "feature5": [1, 1.001, 1.002, 1.003],
                "feature6": [1.5, 2.5, 3.6, 4.8],
            }
        )
        data = DataBase(features=feature_matrix)
        filter = HighQualityFeatureFilter(
            null_ratio_threshold=0.5, dominance_threshold=0.5, variance_threshold=0.1
        )
        self.assertIsNone(filter.selected_features_)

        filter.fit(data)
        self.assertIsInstance(filter.selected_features_, pd.Index)
        self.assertListEqual(
            filter.selected_features_.tolist(), ["feature1", "feature3", "feature6"]
        )

        with self.assertRaises(TypeError):
            filter.fit(None)
        with self.assertRaises(TypeError):
            filter.fit("test")
        with self.assertRaises(TypeError):
            filter.fit(1)
        with self.assertRaises(TypeError):
            filter.fit(feature_matrix)
        with self.assertRaises(ValueError):
            filter.fit(data, layer_in="test")
        with self.assertRaises(TypeError):
            filter.fit(data, layer_in="processed")

        feature_matrix = pd.DataFrame({"feature6": [1.5, 2.5, 3.6, 4.8],})
        data = DataBase(features=feature_matrix)
        filter.fit(data)
        self.assertIsInstance(filter.selected_features_, pd.Index)
        self.assertListEqual(filter.selected_features_.tolist(), ["feature6"])

        filter = HighQualityFeatureFilter(
            null_ratio_threshold=0.1, dominance_threshold=0.5, variance_threshold=0.1
        )
        feature_matrix = pd.DataFrame(
            {
                "feature1": [1, np.nan, np.nan, 2],
                "feature2": [1, np.nan, np.nan, np.nan],
                "feature4": [1, 1, 1, 0],
                "feature5": [1, 1.001, 1.002, 1.003],
            }
        )
        data = DataBase(features=feature_matrix)
        with self.assertRaises(RuntimeError):
            filter.fit(data)

    def test_transform(self):
        """
        Test transform method.
        """
        feature_matrix = pd.DataFrame(
            {
                "feature1": [1, np.nan, np.nan, 2],
                "feature2": [1, np.nan, np.nan, np.nan],
                "feature3": [1, 1, 0, 0],
                "feature4": [1, 1, 1, 0],
                "feature5": [1, 1.001, 1.002, 1.003],
                "feature6": [1.5, 2.5, 3.6, 4.8],
            }
        )
        data = DataBase(features=feature_matrix)
        filter = HighQualityFeatureFilter(
            null_ratio_threshold=0.5, dominance_threshold=0.5, variance_threshold=0.1
        )

        with self.assertRaises(ValueError):
            filter.transform(data)

        filter.fit(data)
        with self.assertRaises(TypeError):
            filter.transform(data, layer_in="processed")
        transformed_data = filter.transform(data, layer_in="raw", layer_out="processed")
        self.assertIsInstance(transformed_data, DataBase)
        raw_feature = transformed_data.get_features(layer="raw")
        transformed_feature = transformed_data.get_features(layer="processed")
        self.assertTrue(raw_feature.equals(feature_matrix))
        self.assertTrue(
            transformed_feature.equals(
                feature_matrix[["feature1", "feature3", "feature6"]]
            )
        )

        with self.assertRaises(TypeError):
            filter.transform(None)
        with self.assertRaises(TypeError):
            filter.transform("test")
        with self.assertRaises(TypeError):
            filter.transform(1)
        with self.assertRaises(TypeError):
            filter.transform(feature_matrix)
        with self.assertRaises(ValueError):
            filter.transform(data, layer_in="test")
        with self.assertRaises(ValueError):
            filter.transform(data, layer_out="test")
        with self.assertRaises(AssertionError):
            filter.transform(DataBase(features=pd.DataFrame()))
        with self.assertRaises(ValueError):
            filter.transform(
                DataBase(features=pd.DataFrame({"feature1": [1, np.nan, np.nan, 2],}))
            )
        filter.transform(data, layer_in="processed")

    def test_fit_transform(self):
        feature_matrix = pd.DataFrame(
            {
                "feature1": [1, np.nan, np.nan, 2],
                "feature2": [1, np.nan, np.nan, np.nan],
                "feature3": [1, 1, 0, 0],
                "feature4": [1, 1, 1, 0],
                "feature5": [1, 1.001, 1.002, 1.003],
                "feature6": [1.5, 2.5, 3.6, 4.8],
            }
        )
        data = DataBase(features=feature_matrix)
        filter = HighQualityFeatureFilter(
            null_ratio_threshold=0.5, dominance_threshold=0.5, variance_threshold=0.1
        )
        transformed_data = filter.fit_transform(data)
        self.assertIsInstance(filter.selected_features_, pd.Index)
        self.assertListEqual(
            filter.selected_features_.tolist(), ["feature1", "feature3", "feature6"]
        )
        raw_feature = transformed_data.get_features(layer="raw")
        transformed_feature = transformed_data.get_features(layer="processed")
        self.assertTrue(raw_feature.equals(feature_matrix))
        self.assertTrue(
            transformed_feature.equals(
                feature_matrix[["feature1", "feature3", "feature6"]]
            )
        )

        with self.assertRaises(TypeError):
            filter.fit_transform(None)
        with self.assertRaises(TypeError):
            filter.fit_transform("test")
        with self.assertRaises(TypeError):
            filter.fit_transform(1)
        with self.assertRaises(TypeError):
            filter.fit_transform(feature_matrix)
        with self.assertRaises(ValueError):
            filter.fit_transform(data, layer_in="test")
        with self.assertRaises(AssertionError):
            filter.fit_transform(DataBase(features=pd.DataFrame()))

        feature_matrix = pd.DataFrame({"feature6": [1.5, 2.5, 3.6, 4.8],})
        data = DataBase(features=feature_matrix)
        transformed_data = filter.fit_transform(data)
        self.assertIsInstance(filter.selected_features_, pd.Index)
        self.assertListEqual(filter.selected_features_.tolist(), ["feature6"])
        raw_feature = transformed_data.get_features(layer="raw")
        transformed_feature = transformed_data.get_features(layer="processed")
        self.assertTrue(raw_feature.equals(feature_matrix))
        self.assertTrue(transformed_feature.equals(feature_matrix))

        filter.fit_transform(data, layer_in="processed")

        filter = HighQualityFeatureFilter(
            null_ratio_threshold=0.1, dominance_threshold=0.5, variance_threshold=0.1
        )
        feature_matrix = pd.DataFrame(
            {
                "feature1": [1, np.nan, np.nan, 2],
                "feature2": [1, np.nan, np.nan, np.nan],
                "feature4": [1, 1, 1, 0],
                "feature5": [1, 1.001, 1.002, 1.003],
            }
        )
        data = DataBase(features=feature_matrix)
        with self.assertRaises(RuntimeError):
            filter.fit_transform(data)


class TestInformativeFeatureFilter(unittest.TestCase):
    """
    Test InformativeFeatureFilter class.
    """

    def setUp(self) -> None:
        """
        """
        pass

    def test_init(self):
        """
        Test __init__ method.
        """
        # Test with default values.
        filter = InformativeFeatureFilter()
        self.assertIsInstance(filter, InformativeFeatureFilter)
        self.assertIsNone(filter.target_corr_threshold)
        self.assertIsNone(filter.chi2_p_threshold)
        self.assertIsNone(filter.fold_change_threshold)

        # Test with specified values.
        filter = InformativeFeatureFilter(
            target_corr_threshold=0.1, chi2_p_threshold=0.5, fold_change_threshold=1.5
        )
        self.assertIsInstance(filter, InformativeFeatureFilter)
        self.assertEqual(filter.target_corr_threshold, 0.1)
        self.assertEqual(filter.chi2_p_threshold, 0.5)
        self.assertEqual(filter.fold_change_threshold, 1.5)

        # Test with invalid values.
        with self.assertRaises(TypeError):
            filter = InformativeFeatureFilter(target_corr_threshold="1")
        with self.assertRaises(TypeError):
            filter = InformativeFeatureFilter(chi2_p_threshold="0.5")
        with self.assertRaises(TypeError):
            filter = InformativeFeatureFilter(fold_change_threshold="-1")

        with self.assertRaises(ValueError):
            filter = InformativeFeatureFilter(target_corr_threshold=-1)
        with self.assertRaises(ValueError):
            filter = InformativeFeatureFilter(target_corr_threshold=1.1)
        with self.assertRaises(ValueError):
            filter = InformativeFeatureFilter(chi2_p_threshold=-1)
        with self.assertRaises(ValueError):
            filter = InformativeFeatureFilter(chi2_p_threshold=1.1)
        with self.assertRaises(ValueError):
            filter = InformativeFeatureFilter(fold_change_threshold=-1)
        with self.assertRaises(ValueError):
            filter = InformativeFeatureFilter(fold_change_threshold=0.5)

        # Test with valid values.
        filter = InformativeFeatureFilter(target_corr_threshold=0)
        self.assertEqual(filter.target_corr_threshold, 0)
        filter = InformativeFeatureFilter(target_corr_threshold=1)
        self.assertEqual(filter.target_corr_threshold, 1)

        filter = InformativeFeatureFilter(chi2_p_threshold=0)
        self.assertEqual(filter.chi2_p_threshold, 0)
        filter = InformativeFeatureFilter(chi2_p_threshold=1)
        self.assertEqual(filter.chi2_p_threshold, 1)

        filter = InformativeFeatureFilter(fold_change_threshold=1)
        self.assertEqual(filter.fold_change_threshold, 1)
        filter = InformativeFeatureFilter(fold_change_threshold=10000)
        self.assertEqual(filter.fold_change_threshold, 10000)

    def test_identify_informative_features(self):
        """
        Test identify_informative_features method.
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

        # Create an instance of InformativeFeatureFilter
        filter = InformativeFeatureFilter()
        selected_features = filter._identify_informative_features(X, y)
        self.assertListEqual(
            selected_features.tolist(),
            ["feature1", "feature2", "feature3", "feature4", "feature5", "feature6"],
        )

        with self.assertRaises(TypeError):
            filter._identify_informative_features(None, y)
        with self.assertRaises(TypeError):
            filter._identify_informative_features("test", y)
        with self.assertRaises(TypeError):
            filter._identify_informative_features(1, y)
        with self.assertRaises(TypeError):
            filter._identify_informative_features(X, None)
        with self.assertRaises(TypeError):
            filter._identify_informative_features(X, "test")
        with self.assertRaises(TypeError):
            filter._identify_informative_features(X, 1)
        with self.assertRaises(ValueError):
            filter._identify_informative_features(X, pd.Series([0, 1, 0, 1]))

        filter = InformativeFeatureFilter(target_corr_threshold=1)
        selected_features = filter._identify_informative_features(X, y)
        self.assertListEqual(
            selected_features.tolist(),
            ["feature1", "feature2", "feature3", "feature4", "feature5", "feature6"],
        )

        filter = InformativeFeatureFilter(chi2_p_threshold=0.00001)
        selected_features = filter._identify_informative_features(X, y)
        self.assertListEqual(
            selected_features.tolist(), ["feature4", "feature5", "feature6"]
        )

        filter = InformativeFeatureFilter(chi2_p_threshold=0.5)
        selected_features = filter._identify_informative_features(X, y)
        self.assertListEqual(
            selected_features.tolist(),
            ["feature1", "feature2", "feature4", "feature5", "feature6"],
        )

        filter = InformativeFeatureFilter(chi2_p_threshold=1)
        selected_features = filter._identify_informative_features(X, y)
        self.assertListEqual(
            selected_features.tolist(),
            ["feature1", "feature2", "feature3", "feature4", "feature5", "feature6"],
        )

        filter = InformativeFeatureFilter(fold_change_threshold=1)
        selected_features = filter._identify_informative_features(X, y)
        self.assertListEqual(
            selected_features.tolist(),
            ["feature1", "feature2", "feature3", "feature4", "feature5", "feature6"],
        )

        filter = InformativeFeatureFilter(fold_change_threshold=10)
        selected_features = filter._identify_informative_features(X, y)
        self.assertListEqual(
            selected_features.tolist(),
            ["feature1", "feature2", "feature3", "feature5", "feature6"],
        )

        filter = InformativeFeatureFilter(fold_change_threshold=1000)
        selected_features = filter._identify_informative_features(X, y)
        self.assertListEqual(
            selected_features.tolist(), ["feature1", "feature2", "feature3"]
        )

        filter = InformativeFeatureFilter(
            chi2_p_threshold=0.00001,
            fold_change_threshold=1000,
            target_corr_threshold=1,
        )
        selected_features = filter._identify_informative_features(X, y)
        self.assertListEqual(selected_features.tolist(), [])

        filter = InformativeFeatureFilter(
            chi2_p_threshold=0.5, fold_change_threshold=10, target_corr_threshold=1,
        )
        selected_features = filter._identify_informative_features(X, y)
        self.assertListEqual(
            selected_features.tolist(), ["feature1", "feature2", "feature5", "feature6"]
        )

        X = pd.DataFrame(
            {
                "feature1": [0, 1, 0, 1, 0, np.nan],
                "feature2": [1, 0, 1, 0, 1, np.nan],
                "feature3": [0, 0, 1, 1, 0, np.nan],
                "feature4": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan],
                "feature5": [0.1, 10, 0.2, 5, 0.3, np.nan],
                "feature6": [-10, 0.1, -5, 0.2, -20, np.nan],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0, 1])
        filter = InformativeFeatureFilter(
            chi2_p_threshold=0.00001,
            fold_change_threshold=1000,
            target_corr_threshold=1,
        )
        selected_features = filter._identify_informative_features(X, y)
        self.assertListEqual(selected_features.tolist(), [])

        filter = InformativeFeatureFilter(
            chi2_p_threshold=0.5, fold_change_threshold=10, target_corr_threshold=1,
        )
        selected_features = filter._identify_informative_features(X, y)
        self.assertListEqual(
            selected_features.tolist(), ["feature1", "feature3", "feature5", "feature6"]
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

        filter = InformativeFeatureFilter()
        selected_features = filter._identify_informative_features(X, y)
        self.assertListEqual(
            selected_features.tolist(),
            ["feature1", "feature2", "feature3", "feature4", "feature5", "feature6"],
        )

        filter = InformativeFeatureFilter(
            chi2_p_threshold=0.00001, fold_change_threshold=1000,
        )
        selected_features = filter._identify_informative_features(X, y)
        self.assertListEqual(
            selected_features.tolist(),
            ["feature1", "feature2", "feature3", "feature4", "feature5", "feature6"],
        )

        filter = InformativeFeatureFilter(target_corr_threshold=0)
        selected_features = filter._identify_informative_features(X, y)
        self.assertListEqual(
            selected_features.tolist(),
            ["feature1", "feature2", "feature3", "feature4", "feature5", "feature6"],
        )

        filter = InformativeFeatureFilter(target_corr_threshold=0.5)
        selected_features = filter._identify_informative_features(X, y)
        self.assertListEqual(
            selected_features.tolist(),
            ["feature1", "feature2", "feature5", "feature6"],
        )

        filter = InformativeFeatureFilter(target_corr_threshold=1)
        selected_features = filter._identify_informative_features(X, y)
        self.assertListEqual(
            selected_features.tolist(), ["feature5"],
        )

        X = pd.DataFrame(
            {
                "feature1": [0, 1, 0, 1, 0, np.nan],
                "feature2": [1, 0, 1, 0, 1, np.nan],
                "feature3": [0, 0, 1, 1, 0, np.nan],
                "feature4": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan],
                "feature5": [0.1, 10, 0.2, 5, 0.3, np.nan],
                "feature6": [-10, 0.1, -5, 0.2, -20, np.nan],
            }
        )
        y = pd.Series([0.1, 10, 0.2, 5, 0.3, 0])
        filter = InformativeFeatureFilter(target_corr_threshold=0.5)
        selected_features = filter._identify_informative_features(X, y)
        self.assertListEqual(
            selected_features.tolist(),
            ["feature1", "feature2", "feature5", "feature6"],
        )

    def test_fit(self):
        """
        Test fit method.
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
        data = DataBase(features=X, labels=y)
        filter = InformativeFeatureFilter()
        self.assertIsNone(filter.identified_features_)

        with self.assertRaises(ValueError):
            filter.fit(data, layer_in="test")
        with self.assertRaises(TypeError):
            filter.fit(data, layer_in="processed")
        filter.fit(data, layer_in="raw")
        self.assertIsInstance(filter.identified_features_, pd.Index)
        self.assertListEqual(
            filter.identified_features_.tolist(),
            ["feature1", "feature2", "feature3", "feature4", "feature5", "feature6"],
        )
        self.assertIsInstance(filter.identified_features_, pd.Index)
        self.assertListEqual(
            filter.identified_features_.tolist(),
            ["feature1", "feature2", "feature3", "feature4", "feature5", "feature6"],
        )

        filter = InformativeFeatureFilter(
            chi2_p_threshold=0.5, fold_change_threshold=10, target_corr_threshold=1,
        )
        filter.fit(data)
        self.assertIsInstance(filter.identified_features_, pd.Index)
        self.assertListEqual(
            filter.identified_features_.tolist(),
            ["feature1", "feature2", "feature5", "feature6"],
        )

        with self.assertRaises(TypeError):
            filter.fit(None)
        with self.assertRaises(TypeError):
            filter.fit("test")
        with self.assertRaises(TypeError):
            filter.fit(1)
        with self.assertRaises(TypeError):
            filter.fit(X)
        with self.assertRaises(TypeError):
            filter.fit(DataBase(features=X))
        filter.fit(DataBase(features=X, labels=pd.Series([0, 1, 0, 1])))

        X = pd.DataFrame(
            {
                "feature1": [0, 1, 0, 1, 0, np.nan],
                "feature2": [1, 0, 1, 0, 1, np.nan],
                "feature3": [0, 0, 1, 1, 0, np.nan],
                "feature4": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan],
                "feature5": [0.1, 10, 0.2, 5, 0.3, np.nan],
                "feature6": [-10, 0.1, -5, 0.2, -20, np.nan],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0, 1])
        data = DataBase(features=X, labels=y)
        filter = InformativeFeatureFilter(
            chi2_p_threshold=0.5, fold_change_threshold=10, target_corr_threshold=1,
        )
        filter.fit(data)
        self.assertIsInstance(filter.identified_features_, pd.Index)
        self.assertListEqual(
            filter.identified_features_.tolist(),
            ["feature1", "feature3", "feature5", "feature6"],
        )

        X = pd.DataFrame(
            {
                "feature1": [0, 1, 0, 1, 0, np.nan],
                "feature2": [1, 0, 1, 0, 1, np.nan],
                "feature3": [0, 0, 1, 1, 0, np.nan],
                "feature4": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan],
                "feature5": [0.1, 10, 0.2, 5, 0.3, np.nan],
                "feature6": [-10, 0.1, -5, 0.2, -20, np.nan],
            }
        )
        y = pd.Series([0.1, 10, 0.2, 5, 0.3, 0])
        filter = InformativeFeatureFilter(target_corr_threshold=0.5)
        filter.fit(DataBase(features=X, labels=y))
        self.assertIsInstance(filter.identified_features_, pd.Index)
        self.assertListEqual(
            filter.identified_features_.tolist(),
            ["feature1", "feature2", "feature5", "feature6"],
        )

        X = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [-10, 0.1, -5, 0.2, -20],
            }
        )
        y = pd.Series([0.1, 10, 0.2, 5, 0.3, 0])
        data = DataBase(features=X, labels=y)
        filter = InformativeFeatureFilter(target_corr_threshold=1)
        with self.assertRaises(RuntimeError):
            filter.fit(data)

    def test_transform(self):
        """
        Test transform method.
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
        data = DataBase(features=X, labels=y)
        filter = InformativeFeatureFilter()

        with self.assertRaises(ValueError):
            filter.transform(data)
        with self.assertRaises(ValueError):
            filter.transform(data, layer_in="processed")
        filter.fit(data, layer_in="raw")
        processed_data = filter.transform(data, layer_in="raw")
        self.assertIsInstance(processed_data, DataBase)
        raw_feature = processed_data.get_features(layer="raw")
        processed_feature = processed_data.get_features(layer="processed")

        self.assertTrue(raw_feature.equals(X))
        self.assertTrue(processed_feature.equals(X))

        filter = InformativeFeatureFilter(
            chi2_p_threshold=0.5, fold_change_threshold=10, target_corr_threshold=1,
        )
        filter.fit(data)
        processed_data = filter.transform(data)
        self.assertIsInstance(processed_data, DataBase)
        raw_feature = processed_data.get_features(layer="raw")
        processed_feature = processed_data.get_features(layer="processed")
        self.assertTrue(raw_feature.equals(X))
        self.assertTrue(
            processed_feature.equals(
                X[["feature1", "feature2", "feature5", "feature6"]]
            )
        )
        with self.assertRaises(ValueError):
            filter.transform(data, layer_in="test")
        with self.assertRaises(ValueError):
            filter.transform(data, layer_out="test")
        with self.assertRaises(TypeError):
            filter.transform(None)
        with self.assertRaises(TypeError):
            filter.transform("test")
        with self.assertRaises(TypeError):
            filter.transform(1)
        with self.assertRaises(TypeError):
            filter.transform(X)
        with self.assertRaises(AssertionError):
            filter.transform(DataBase(features=pd.DataFrame()))
        with self.assertRaises(ValueError):
            filter.transform(
                DataBase(features=pd.DataFrame({"feature1": [0, 1, 0, 1, 0],}))
            )
        with self.assertRaises(ValueError):
            filter.transform(
                DataBase(
                    features=pd.DataFrame({"feature1": [0, 1, 0, 1, 0],}), labels=y
                )
            )
        filter.transform(data, layer_in="processed")

    def test_fit_transform(self):
        """
        Test fit_transform method.
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
        data = DataBase(features=X, labels=y)
        filter = InformativeFeatureFilter()

        with self.assertRaises(ValueError):
            filter.fit_transform(data, layer_in="test")
        with self.assertRaises(TypeError):
            filter.fit_transform(data, layer_in="processed")
        processed_data = filter.fit_transform(data, layer_in="raw")
        self.assertIsInstance(processed_data, DataBase)
        raw_feature = processed_data.get_features(layer="raw")
        processed_feature = processed_data.get_features(layer="processed")

        self.assertTrue(raw_feature.equals(X))
        self.assertTrue(processed_feature.equals(X))

        filter = InformativeFeatureFilter(
            chi2_p_threshold=0.5, fold_change_threshold=10, target_corr_threshold=1,
        )
        processed_data = filter.fit_transform(data)
        self.assertIsInstance(processed_data, DataBase)
        raw_feature = processed_data.get_features(layer="raw")
        processed_feature = processed_data.get_features(layer="processed")
        self.assertTrue(raw_feature.equals(X))
        self.assertTrue(
            processed_feature.equals(
                X[["feature1", "feature2", "feature5", "feature6"]]
            )
        )

        filter.fit_transform(data, layer_in="processed")
        with self.assertRaises(ValueError):
            filter.fit_transform(data, layer_in="test")
        with self.assertRaises(TypeError):
            filter.fit_transform(data, layer_out="processed")
        with self.assertRaises(TypeError):
            filter.fit_transform(None)
        with self.assertRaises(TypeError):
            filter.fit_transform("test")
        with self.assertRaises(TypeError):
            filter.fit_transform(1)
        with self.assertRaises(TypeError):
            filter.fit_transform(X)
        with self.assertRaises(AssertionError):
            filter.fit_transform(DataBase(features=pd.DataFrame()))

        X = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [-10, 0.1, -5, 0.2, -20],
            }
        )
        y = pd.Series([0.1, 10, 0.2, 5, 0.3, 0])
        data = DataBase(features=X, labels=y)
        filter = InformativeFeatureFilter(target_corr_threshold=1)
        with self.assertRaises(RuntimeError):
            filter.fit_transform(data)


class TestMissingValueImputer(unittest.TestCase):
    """
    Test MissingValueImputer class.
    """

    def setUp(self) -> None:
        """
        """
        pass

    def test_init(self):
        """
        Test __init__ method.
        """
        # Test with default values.
        imputer = MissingValueImputer()
        self.assertIsInstance(imputer, MissingValueImputer)
        self.assertEqual(imputer.impute_strategy, "mean")
        self.assertIsNone(imputer.impute_fill_value)
        self.assertIsNone(imputer.impute_values_)

        # Test with specified values.
        imputer = MissingValueImputer(impute_strategy="constant", impute_fill_value=0)
        self.assertIsInstance(imputer, MissingValueImputer)
        self.assertEqual(imputer.impute_strategy, "constant")
        self.assertEqual(imputer.impute_fill_value, 0)
        self.assertIsNone(imputer.impute_values_)

        # Test with invalid values.
        with self.assertRaises(ValueError):
            imputer = MissingValueImputer(impute_strategy=1)
        with self.assertRaises(TypeError):
            imputer = MissingValueImputer(impute_fill_value="0")
        with self.assertRaises(ValueError):
            imputer = MissingValueImputer(impute_strategy="test")
        with self.assertRaises(ValueError):
            imputer = MissingValueImputer(impute_strategy="constant")

        # Test with valid values.
        imputer = MissingValueImputer(impute_strategy="mean")
        self.assertEqual(imputer.impute_strategy, "mean")
        imputer = MissingValueImputer(impute_strategy="median")
        self.assertEqual(imputer.impute_strategy, "median")
        imputer = MissingValueImputer(impute_strategy="most_frequent")
        self.assertEqual(imputer.impute_strategy, "most_frequent")
        imputer = MissingValueImputer(impute_strategy="constant", impute_fill_value=0)
        self.assertEqual(imputer.impute_strategy, "constant")
        imputer = MissingValueImputer(impute_fill_value=0)
        self.assertEqual(imputer.impute_fill_value, 0)

    def test_impute_missing_values(self):
        """
        Test impute_missing_values method.
        """
        # Test with default values.
        imputer = MissingValueImputer()

        # Test with invalid values.
        with self.assertRaises(TypeError):
            imputer._impute_missing_values(None)
        with self.assertRaises(TypeError):
            imputer._impute_missing_values("test")
        with self.assertRaises(TypeError):
            imputer._impute_missing_values(1)

        # Test with valid values.
        X = pd.DataFrame(
            {
                "feature1": [1, 1, 1, np.nan, np.nan],
                "feature2": [1, 2, 4, np.nan, 1],
                "feature3": [1, 2, 3, np.nan, np.nan],
                "feature4": [2, 3, 1, np.nan, np.nan],
                "feature5": [1, 1, 1, 1, 1],
                "feature6": [np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )
        imputer = MissingValueImputer(impute_strategy="mean")
        impute_values = imputer._impute_missing_values(X)
        self.assertEqual(
            {key: value for key, value in impute_values.items() if not np.isnan(value)},
            {
                "feature1": 1,
                "feature2": 2,
                "feature3": 2,
                "feature4": 2,
                "feature5": 1,
            },
        )
        self.assertTrue(np.isnan(impute_values["feature6"]))
        imputer = MissingValueImputer(impute_strategy="median")
        impute_values = imputer._impute_missing_values(X)
        self.assertEqual(
            {key: value for key, value in impute_values.items() if not np.isnan(value)},
            {
                "feature1": 1,
                "feature2": 1.5,
                "feature3": 2,
                "feature4": 2,
                "feature5": 1,
            },
        )
        self.assertTrue(np.isnan(impute_values["feature6"]))
        imputer = MissingValueImputer(impute_strategy="most_frequent")
        impute_values = imputer._impute_missing_values(X)
        self.assertEqual(
            {key: value for key, value in impute_values.items() if not np.isnan(value)},
            {
                "feature1": 1,
                "feature2": 1,
                "feature3": 1,
                "feature4": 1,
                "feature5": 1,
            },
        )
        self.assertTrue(np.isnan(impute_values["feature6"]))

        imputer = MissingValueImputer(impute_strategy="constant", impute_fill_value=0)
        impute_values = imputer._impute_missing_values(X)
        self.assertEqual(
            impute_values,
            {
                "feature1": 0,
                "feature2": 0,
                "feature3": 0,
                "feature4": 0,
                "feature5": 0,
                "feature6": 0,
            },
        )

        impute_values = imputer._impute_missing_values(pd.DataFrame())
        self.assertEqual(impute_values, {})

    def test_fit(self):
        """
        Test fit method.
        """
        X = pd.DataFrame(
            {
                "feature1": [1, 1, 1, np.nan, np.nan],
                "feature2": [1, 2, 4, np.nan, 1],
                "feature3": [1, 2, 3, np.nan, np.nan],
                "feature4": [2, 3, 1, np.nan, np.nan],
                "feature5": [1, 1, 1, 1, 1],
                "feature6": [np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )
        data = DataBase(features=X)
        imputer = MissingValueImputer()
        self.assertIsNone(imputer.impute_values_)

        with self.assertRaises(TypeError):
            imputer.fit(None)
        with self.assertRaises(TypeError):
            imputer.fit("test")
        with self.assertRaises(TypeError):
            imputer.fit(1)
        with self.assertRaises(TypeError):
            imputer.fit(X)
        imputer.fit(DataBase(features=X, labels=pd.Series([0, 1, 0, 1])))

        imputer.fit(data)
        self.assertIsInstance(imputer.impute_values_, dict)
        self.assertEqual(
            {
                key: value
                for key, value in imputer.impute_values_.items()
                if not np.isnan(value)
            },
            {
                "feature1": 1,
                "feature2": 2,
                "feature3": 2,
                "feature4": 2,
                "feature5": 1,
            },
        )
        self.assertTrue(np.isnan(imputer.impute_values_["feature6"]))

        imputer = MissingValueImputer(impute_strategy="median")
        imputer.fit(data)
        self.assertIsInstance(imputer.impute_values_, dict)
        self.assertEqual(
            {
                key: value
                for key, value in imputer.impute_values_.items()
                if not np.isnan(value)
            },
            {
                "feature1": 1,
                "feature2": 1.5,
                "feature3": 2,
                "feature4": 2,
                "feature5": 1,
            },
        )
        self.assertTrue(np.isnan(imputer.impute_values_["feature6"]))

        imputer = MissingValueImputer(impute_strategy="most_frequent")
        imputer.fit(data)
        self.assertIsInstance(imputer.impute_values_, dict)
        self.assertEqual(
            {
                key: value
                for key, value in imputer.impute_values_.items()
                if not np.isnan(value)
            },
            {
                "feature1": 1,
                "feature2": 1,
                "feature3": 1,
                "feature4": 1,
                "feature5": 1,
            },
        )
        self.assertTrue(np.isnan(imputer.impute_values_["feature6"]))

        imputer = MissingValueImputer(impute_strategy="constant", impute_fill_value=0)
        imputer.fit(data)
        self.assertIsInstance(imputer.impute_values_, dict)
        self.assertEqual(
            imputer.impute_values_,
            {
                "feature1": 0,
                "feature2": 0,
                "feature3": 0,
                "feature4": 0,
                "feature5": 0,
                "feature6": 0,
            },
        )

        with self.assertRaises(ValueError):
            imputer.fit(data, layer_in="test")
        with self.assertRaises(TypeError):
            imputer.fit(data, layer_in="processed")

    def test_transform(self):
        """
        Test transform method.
        """
        X = pd.DataFrame(
            {
                "feature1": [1, 1, 1, np.nan, np.nan],
                "feature2": [1, 2, 4, np.nan, 1],
                "feature3": [1, 2, 3, np.nan, np.nan],
                "feature4": [2, 3, 1, np.nan, np.nan],
                "feature5": [1, 1, 1, 1, 1],
                "feature6": [np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )
        data = DataBase(features=X)
        imputer = MissingValueImputer()

        with self.assertRaises(ValueError):
            imputer.transform(data)

        imputer.fit(data)
        with self.assertRaises(ValueError):
            imputer.transform(data, layer_in="processed")

        processed_data = imputer.transform(data, layer_in="raw")
        self.assertIsInstance(processed_data, DataBase)
        raw_feature = processed_data.get_features(layer="raw")
        processed_feature = processed_data.get_features(layer="processed").astype(float)
        target_feature = pd.DataFrame(
            {
                "feature1": [1, 1, 1, 1, 1],
                "feature2": [1, 2, 4, 2, 1],
                "feature3": [1, 2, 3, 2, 2],
                "feature4": [2, 3, 1, 2, 2],
                "feature5": [1, 1, 1, 1, 1],
                "feature6": [np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        ).astype(float)
        self.assertTrue(raw_feature.equals(X))
        self.assertTrue(processed_feature.equals(target_feature))

        imputer = MissingValueImputer(impute_strategy="median")
        imputer.fit(data)
        processed_data = imputer.transform(data)
        self.assertIsInstance(processed_data, DataBase)
        raw_feature = processed_data.get_features(layer="raw")
        processed_feature = processed_data.get_features(layer="processed").astype(float)
        target_feature = pd.DataFrame(
            {
                "feature1": [1, 1, 1, 1, 1],
                "feature2": [1, 2, 4, 1.5, 1],
                "feature3": [1, 2, 3, 2, 2],
                "feature4": [2, 3, 1, 2, 2],
                "feature5": [1, 1, 1, 1, 1],
                "feature6": [np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        ).astype(float)
        self.assertTrue(raw_feature.equals(X))
        self.assertTrue(processed_feature.equals(target_feature))

        imputer = MissingValueImputer(impute_strategy="most_frequent")
        imputer.fit(data)
        processed_data = imputer.transform(data)
        self.assertIsInstance(processed_data, DataBase)
        raw_feature = processed_data.get_features(layer="raw")
        processed_feature = processed_data.get_features(layer="processed").astype(float)
        target_feature = pd.DataFrame(
            {
                "feature1": [1, 1, 1, 1, 1],
                "feature2": [1, 2, 4, 1, 1],
                "feature3": [1, 2, 3, 1, 1],
                "feature4": [2, 3, 1, 1, 1],
                "feature5": [1, 1, 1, 1, 1],
                "feature6": [np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        ).astype(float)
        self.assertTrue(raw_feature.equals(X))
        self.assertTrue(processed_feature.equals(target_feature))

        imputer = MissingValueImputer(impute_strategy="constant", impute_fill_value=0)
        imputer.fit(data)
        processed_data = imputer.transform(data)
        self.assertIsInstance(processed_data, DataBase)
        raw_feature = processed_data.get_features(layer="raw")
        processed_feature = processed_data.get_features(layer="processed").astype(float)
        target_feature = pd.DataFrame(
            {
                "feature1": [1, 1, 1, 0, 0],
                "feature2": [1, 2, 4, 0, 1],
                "feature3": [1, 2, 3, 0, 0],
                "feature4": [2, 3, 1, 0, 0],
                "feature5": [1, 1, 1, 1, 1],
                "feature6": [0, 0, 0, 0, 0],
            }
        ).astype(float)
        self.assertTrue(raw_feature.equals(X))
        self.assertTrue(processed_feature.equals(target_feature))

        with self.assertRaises(ValueError):
            imputer.transform(data, layer_out="test")
        with self.assertRaises(TypeError):
            imputer.transform(None)
        with self.assertRaises(TypeError):
            imputer.transform("test")
        with self.assertRaises(TypeError):
            imputer.transform(1)
        with self.assertRaises(TypeError):
            imputer.transform(X)
        imputer.transform(DataBase(features=X, labels=pd.Series([0, 1, 0, 1])))

        with self.assertRaises(AssertionError):
            processed_data = imputer.transform(DataBase(features=pd.DataFrame()))

        processed_data = imputer.transform(data, layer_in="processed")
        self.assertIsInstance(processed_data, DataBase)
        processed_feature = processed_data.get_features(layer="processed").astype(float)
        target_feature = pd.DataFrame(
            {
                "feature1": [1, 1, 1, 0, 0],
                "feature2": [1, 2, 4, 0, 1],
                "feature3": [1, 2, 3, 0, 0],
                "feature4": [2, 3, 1, 0, 0],
                "feature5": [1, 1, 1, 1, 1],
                "feature6": [0, 0, 0, 0, 0],
            }
        ).astype(float)
        self.assertTrue(processed_feature.equals(target_feature))

        with self.assertRaises(ValueError):
            imputer.transform(
                DataBase(
                    features=pd.DataFrame({"feature1": [1, 1, 1, np.nan, np.nan],})
                )
            )
        imputer.transform(data, layer_in="processed")

    def test_fit_transform(self):
        """
        Test fit_transform method.
        """
        X = pd.DataFrame(
            {
                "feature1": [1, 1, 1, np.nan, np.nan],
                "feature2": [1, 2, 4, np.nan, 1],
                "feature3": [1, 2, 3, np.nan, np.nan],
                "feature4": [2, 3, 1, np.nan, np.nan],
                "feature5": [1, 1, 1, 1, 1],
                "feature6": [np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )
        data = DataBase(features=X)
        imputer = MissingValueImputer()

        with self.assertRaises(TypeError):
            imputer.fit_transform(data, layer_in="processed")
        with self.assertRaises(ValueError):
            imputer.fit_transform(data, layer_in="test")
        processed_data = imputer.fit_transform(data, layer_in="raw")
        self.assertIsInstance(processed_data, DataBase)
        raw_feature = processed_data.get_features(layer="raw")
        processed_feature = processed_data.get_features(layer="processed").astype(float)
        target_feature = pd.DataFrame(
            {
                "feature1": [1, 1, 1, 1, 1],
                "feature2": [1, 2, 4, 2, 1],
                "feature3": [1, 2, 3, 2, 2],
                "feature4": [2, 3, 1, 2, 2],
                "feature5": [1, 1, 1, 1, 1],
                "feature6": [np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        ).astype(float)
        self.assertTrue(raw_feature.equals(X))
        self.assertTrue(processed_feature.equals(target_feature))

        imputer = MissingValueImputer(impute_strategy="median")
        processed_data = imputer.fit_transform(data)
        self.assertIsInstance(processed_data, DataBase)
        raw_feature = processed_data.get_features(layer="raw")
        processed_feature = processed_data.get_features(layer="processed").astype(float)
        target_feature = pd.DataFrame(
            {
                "feature1": [1, 1, 1, 1, 1],
                "feature2": [1, 2, 4, 1.5, 1],
                "feature3": [1, 2, 3, 2, 2],
                "feature4": [2, 3, 1, 2, 2],
                "feature5": [1, 1, 1, 1, 1],
                "feature6": [np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        ).astype(float)
        self.assertTrue(raw_feature.equals(X))
        self.assertTrue(processed_feature.equals(target_feature))

        imputer = MissingValueImputer(impute_strategy="most_frequent")
        processed_data = imputer.fit_transform(data)
        self.assertIsInstance(processed_data, DataBase)
        raw_feature = processed_data.get_features(layer="raw")
        processed_feature = processed_data.get_features(layer="processed").astype(float)
        target_feature = pd.DataFrame(
            {
                "feature1": [1, 1, 1, 1, 1],
                "feature2": [1, 2, 4, 1, 1],
                "feature3": [1, 2, 3, 1, 1],
                "feature4": [2, 3, 1, 1, 1],
                "feature5": [1, 1, 1, 1, 1],
                "feature6": [np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        ).astype(float)
        self.assertTrue(raw_feature.equals(X))
        self.assertTrue(processed_feature.equals(target_feature))

        imputer = MissingValueImputer(impute_strategy="constant", impute_fill_value=0)
        processed_data = imputer.fit_transform(data)
        self.assertIsInstance(processed_data, DataBase)
        raw_feature = processed_data.get_features(layer="raw")
        processed_feature = processed_data.get_features(layer="processed").astype(float)
        target_feature = pd.DataFrame(
            {
                "feature1": [1, 1, 1, 0, 0],
                "feature2": [1, 2, 4, 0, 1],
                "feature3": [1, 2, 3, 0, 0],
                "feature4": [2, 3, 1, 0, 0],
                "feature5": [1, 1, 1, 1, 1],
                "feature6": [0, 0, 0, 0, 0],
            }
        ).astype(float)
        self.assertTrue(raw_feature.equals(X))
        self.assertTrue(processed_feature.equals(target_feature))

        with self.assertRaises(TypeError):
            imputer.fit_transform(None)
        with self.assertRaises(TypeError):
            imputer.fit_transform("test")
        with self.assertRaises(TypeError):
            imputer.fit_transform(1)
        with self.assertRaises(TypeError):
            imputer.fit_transform(X)
        imputer.fit_transform(DataBase(features=X, labels=pd.Series([0, 1, 0, 1])))
        imputer.fit_transform(data, layer_in="processed")


class TestSkewnessCorrector(unittest.TestCase):
    """
    Test SkewnessCorrector class.
    """

    def setUp(self) -> None:
        """
        """
        pass

    def test_init(self):
        """
        Test __init__ method.
        """
        # Test with default values.
        corrector = SkewnessCorrector()
        self.assertIsInstance(corrector, SkewnessCorrector)
        self.assertEqual(corrector.skew_correction_method, "box-cox")
        self.assertEqual(corrector.skew_threshold, 0.5)
        self.assertIsNone(corrector.left_skewed_features_)
        self.assertIsNone(corrector.right_skewed_features_)

        # Test with specified values.
        corrector = SkewnessCorrector(skew_correction_method="exp")
        self.assertIsInstance(corrector, SkewnessCorrector)
        self.assertEqual(corrector.skew_correction_method, "exp")
        self.assertEqual(corrector.skew_threshold, 0.5)
        self.assertIsNone(corrector.left_skewed_features_)
        self.assertIsNone(corrector.right_skewed_features_)

        corrector = SkewnessCorrector(skew_correction_method="square")
        self.assertIsInstance(corrector, SkewnessCorrector)
        self.assertEqual(corrector.skew_correction_method, "square")
        self.assertEqual(corrector.skew_threshold, 0.5)

        corrector = SkewnessCorrector(skew_correction_method="yeo-johnson")
        self.assertIsInstance(corrector, SkewnessCorrector)
        self.assertEqual(corrector.skew_correction_method, "yeo-johnson")
        self.assertEqual(corrector.skew_threshold, 0.5)

        corrector = SkewnessCorrector(skew_threshold=100)
        self.assertEqual(corrector.skew_threshold, 100)

        with self.assertRaises(ValueError):
            corrector = SkewnessCorrector(skew_correction_method="test")
        with self.assertRaises(ValueError):
            corrector = SkewnessCorrector(skew_threshold=-1)
        with self.assertRaises(TypeError):
            corrector = SkewnessCorrector(skew_threshold="test")
        with self.assertRaises(ValueError):
            corrector = SkewnessCorrector(skew_threshold=0)

    def test_identify_skewed_features(self):
        """
        Test _identify_skewed_features method.
        """
        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, -1000],
                "feature2": [4, 5, 5, 5, 5, 5, 5, 10000],
                "feature3": [1, 2, 3, 4, 5, 6, 7, 8],
                "feature4": [100, 100, 100, 100, 100, 100, 100, 1],
                "feature5": [100, 100, 100, 100, 100, 100, np.nan, 1],
                "feature6": [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
            }
        )
        corrector = SkewnessCorrector()
        with self.assertRaises(TypeError):
            corrector._identify_skewed_features(None)
        with self.assertRaises(TypeError):
            corrector._identify_skewed_features("test")
        with self.assertRaises(TypeError):
            corrector._identify_skewed_features(1)

        (
            left_skewed_features,
            right_skewed_features,
        ) = corrector._identify_skewed_features(X)
        self.assertListEqual(left_skewed_features, ["feature4"])
        self.assertListEqual(right_skewed_features, ["feature2"])

        (
            left_skewed_features,
            right_skewed_features,
        ) = corrector._identify_skewed_features(pd.DataFrame())
        self.assertListEqual(left_skewed_features, [])
        self.assertListEqual(right_skewed_features, [])

        corrector = SkewnessCorrector(skew_threshold=10)
        (
            left_skewed_features,
            right_skewed_features,
        ) = corrector._identify_skewed_features(X)
        self.assertListEqual(
            left_skewed_features, [],
        )
        self.assertListEqual(
            right_skewed_features, [],
        )

    def test_fit(self):
        """
        Test fit method.
        """
        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, -1000],
                "feature2": [4, 5, 5, 5, 5, 5, 5, 10000],
                "feature3": [1, 2, 3, 4, 5, 6, 7, 8],
                "feature4": [100, 100, 100, 100, 100, 100, 100, 1],
                "feature5": [100, 100, 100, 100, 100, 100, np.nan, 1],
                "feature6": [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
            }
        )
        data = DataBase(features=X)
        corrector = SkewnessCorrector()
        self.assertIsNone(corrector.left_skewed_features_)
        self.assertIsNone(corrector.right_skewed_features_)

        with self.assertRaises(TypeError):
            corrector.fit(data=None)
        with self.assertRaises(TypeError):
            corrector.fit(data="test")
        with self.assertRaises(TypeError):
            corrector.fit(data=1)
        with self.assertRaises(TypeError):
            corrector.fit(data=X)
        corrector.fit(DataBase(features=X, labels=pd.Series([0, 1, 0, 1])))
        with self.assertRaises(AssertionError):
            corrector.fit(DataBase(features=pd.DataFrame()))
        with self.assertRaises(ValueError):
            corrector.fit(data=data, layer_in="test")
        with self.assertRaises(TypeError):
            corrector.fit(data=data, layer_in="processed")

        corrector.fit(data)
        self.assertListEqual(corrector.left_skewed_features_, ["feature4"])
        self.assertListEqual(corrector.right_skewed_features_, ["feature2"])

        corrector.fit(data)
        self.assertListEqual(corrector.left_skewed_features_, ["feature4"])
        self.assertListEqual(corrector.right_skewed_features_, ["feature2"])

        corrector = SkewnessCorrector(skew_threshold=10)
        corrector.fit(data)
        self.assertListEqual(corrector.left_skewed_features_, [])
        self.assertListEqual(corrector.right_skewed_features_, [])

    def test_transform(self):
        """
        Test transform method.
        """
        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, -1000],
                "feature2": [4, 5, 5, 5, 5, 5, 5, 10000],
                "feature3": [1, 2, 3, 4, 5, 6, 7, 8],
                "feature4": [100, 100, 100, 100, 100, 100, 100, 1],
                "feature5": [100, 100, 100, 100, 100, 100, np.nan, 1],
                "feature6": [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
            }
        ).astype(float)
        data = DataBase(features=X)
        corrector = SkewnessCorrector()

        with self.assertRaises(ValueError):
            corrector.transform(data)

        corrector.fit(data)
        with self.assertRaises(ValueError):
            corrector.transform(data, layer_in="processed")

        processed_data = corrector.transform(data, layer_in="raw")
        self.assertIsInstance(processed_data, DataBase)
        raw_feature = processed_data.get_features(layer="raw")
        processed_feature = processed_data.get_features(layer="processed").astype(float)
        target_feature = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, -1000],
                "feature2": np.exp(X["feature2"]),
                "feature3": [1, 2, 3, 4, 5, 6, 7, 8],
                "feature4": np.log(X["feature4"]),
                "feature5": X["feature5"],
                "feature6": [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
            }
        ).astype(float)
        self.assertTrue(raw_feature.equals(X))
        target_feature[target_feature > np.exp(5)] = np.exp(5)
        self.assertTrue(processed_feature.equals(target_feature))

        with self.assertRaises(ValueError):
            corrector.transform(
                DataBase(
                    features=pd.DataFrame({"feature1": [1, 2, 3, 4, 5, 6, 7, -1000],})
                )
            )
        corrector.transform(data, layer_in="processed")

    def test_fit_transform(self):
        """
        Test fit_transform method.
        """
        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, -1000],
                "feature2": [4, 5, 5, 5, 5, 5, 5, 10000],
                "feature3": [1, 2, 3, 4, 5, 6, 7, 8],
                "feature4": [100, 100, 100, 100, 100, 100, 100, 1],
                "feature5": [100, 100, 100, 100, 100, 100, np.nan, 1],
                "feature6": [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
            }
        ).astype(float)
        data = DataBase(features=X)
        corrector = SkewnessCorrector()

        with self.assertRaises(TypeError):
            corrector.fit_transform(data, layer_in="processed")
        with self.assertRaises(ValueError):
            corrector.fit_transform(data, layer_in="test")
        processed_data = corrector.fit_transform(data, layer_in="raw")
        self.assertIsInstance(processed_data, DataBase)
        raw_feature = processed_data.get_features(layer="raw")
        processed_feature = processed_data.get_features(layer="processed").astype(float)
        target_feature = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, -1000],
                "feature2": np.exp(X["feature2"]),
                "feature3": [1, 2, 3, 4, 5, 6, 7, 8],
                "feature4": np.log(X["feature4"]),
                "feature5": X["feature5"],
                "feature6": [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
            }
        ).astype(float)
        self.assertTrue(raw_feature.equals(X))
        target_feature[target_feature > np.exp(5)] = np.exp(5)
        self.assertTrue(processed_feature.equals(target_feature))

        with self.assertRaises(TypeError):
            corrector.fit_transform(None)
        with self.assertRaises(TypeError):
            corrector.fit_transform("test")
        with self.assertRaises(TypeError):
            corrector.fit_transform(1)
        with self.assertRaises(TypeError):
            corrector.fit_transform(X)
        corrector.fit_transform(DataBase(features=X, labels=pd.Series([0, 1, 0, 1])))
        with self.assertRaises(AssertionError):
            corrector.fit_transform(DataBase(features=pd.DataFrame()))

        corrector = SkewnessCorrector(skew_threshold=10)
        processed_data = corrector.fit_transform(data)
        self.assertIsInstance(processed_data, DataBase)
        raw_feature = processed_data.get_features(layer="raw")
        processed_feature = processed_data.get_features(layer="processed").astype(float)
        self.assertTrue(raw_feature.equals(X))
        self.assertTrue(processed_feature.equals(X))

        corrector.fit_transform(data, layer_in="processed")


class TestOutlierCorrector(unittest.TestCase):
    """
    Test OutlierCorrector class.
    """

    def setUp(self) -> None:
        """
        """
        pass

    def test_init(self):
        """
        Test __init__ method.
        """
        # Test with default values.
        corrector = OutlierCorrector()
        self.assertIsInstance(corrector, OutlierCorrector)
        self.assertEqual(corrector.detection_method, "z_score")
        self.assertEqual(corrector.detection_threshold, 3)
        self.assertEqual(corrector.correction_method, "clip")
        self.assertIsNone(corrector.outliers_)

        # Test with specified values.
        corrector = OutlierCorrector(
            detection_method="iqr", detection_threshold=1.5, correction_method="mean"
        )
        self.assertIsInstance(corrector, OutlierCorrector)
        self.assertEqual(corrector.detection_method, "iqr")
        self.assertEqual(corrector.detection_threshold, 1.5)
        self.assertEqual(corrector.correction_method, "mean")
        self.assertIsNone(corrector.outliers_)

        corrector = OutlierCorrector(
            detection_method="modified_z_score",
            detection_threshold=10,
            correction_method="median",
        )
        self.assertIsInstance(corrector, OutlierCorrector)
        self.assertEqual(corrector.detection_method, "modified_z_score")
        self.assertEqual(corrector.detection_threshold, 10)
        self.assertEqual(corrector.correction_method, "median")
        self.assertIsNone(corrector.outliers_)

        with self.assertRaises(ValueError):
            corrector = OutlierCorrector(detection_method="test")
        with self.assertRaises(ValueError):
            corrector = OutlierCorrector(detection_threshold=-1)
        with self.assertRaises(TypeError):
            corrector = OutlierCorrector(detection_threshold="test")
        with self.assertRaises(ValueError):
            corrector = OutlierCorrector(detection_threshold=0)
        with self.assertRaises(ValueError):
            corrector = OutlierCorrector(correction_method="test")
        with self.assertRaises(ValueError):
            corrector = OutlierCorrector(
                detection_method="iqr", detection_threshold=0.1
            )

    def test_identify_outliers(self):
        """
        Test _identify_outliers method.
        """
        X = pd.DataFrame(
            {
                "feature1": [100, 100, 100, 100, 100, 100, 100, 1],
                "feature2": [100, 100, 100, 100, 100, 100, np.nan, 1],
                "feature3": [1, 2, 3, 4, 5, 6, 7, 10000],
                "feature4": [1, 2, 3, 4, 5, 6, 7, 8],
                "feature5": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1],
                "feature6": ["a", "b", "c", "d", "e", "f", "g", "h"],
            }
        )
        corrector = OutlierCorrector(detection_method="z_score", detection_threshold=2)
        with self.assertRaises(TypeError):
            corrector._identify_outliers(None)
        with self.assertRaises(TypeError):
            corrector._identify_outliers("test")
        with self.assertRaises(TypeError):
            corrector._identify_outliers(1)

        outliers = corrector._identify_outliers(X)
        target_outliers = {
            "feature1": pd.Series(
                [False, False, False, False, False, False, False, True]
            ),
            "feature2": pd.Series(
                [False, False, False, False, False, False, False, True]
            ),
            "feature3": pd.Series(
                [False, False, False, False, False, False, False, True]
            ),
            "feature4": pd.Series(
                [False, False, False, False, False, False, False, False]
            ),
            "feature5": pd.Series(
                [False, False, False, False, False, False, False, False]
            ),
        }
        self.assertEqual(outliers.keys(), target_outliers.keys())
        for key, value in outliers.items():
            self.assertTrue(value.equals(target_outliers[key]))

        corrector = OutlierCorrector(detection_method="z_score", detection_threshold=1)
        outliers = corrector._identify_outliers(X)
        target_outliers = {
            "feature1": pd.Series(
                [False, False, False, False, False, False, False, True]
            ),
            "feature2": pd.Series(
                [False, False, False, False, False, False, False, True]
            ),
            "feature3": pd.Series(
                [False, False, False, False, False, False, False, True]
            ),
            "feature4": pd.Series([True, True, False, False, False, False, True, True]),
            "feature5": pd.Series(
                [False, False, False, False, False, False, False, False]
            ),
        }
        self.assertEqual(outliers.keys(), target_outliers.keys())
        for key, value in outliers.items():
            self.assertTrue(value.equals(target_outliers[key]))

        corrector = OutlierCorrector(detection_method="z_score", detection_threshold=10)
        outliers = corrector._identify_outliers(X)
        target_outliers = {
            "feature1": pd.Series(
                [False, False, False, False, False, False, False, False]
            ),
            "feature2": pd.Series(
                [False, False, False, False, False, False, False, False]
            ),
            "feature3": pd.Series(
                [False, False, False, False, False, False, False, False]
            ),
            "feature4": pd.Series(
                [False, False, False, False, False, False, False, False]
            ),
            "feature5": pd.Series(
                [False, False, False, False, False, False, False, False]
            ),
        }
        self.assertEqual(outliers.keys(), target_outliers.keys())
        for key, value in outliers.items():
            self.assertTrue(value.equals(target_outliers[key]))

        corrector = OutlierCorrector(detection_method="iqr", detection_threshold=1.1)
        outliers = corrector._identify_outliers(X)
        target_outliers = {
            "feature1": pd.Series(
                [False, False, False, False, False, False, False, True]
            ),
            "feature2": pd.Series(
                [False, False, False, False, False, False, False, True]
            ),
            "feature3": pd.Series(
                [False, False, False, False, False, False, False, True]
            ),
            "feature4": pd.Series(
                [False, False, False, False, False, False, False, False]
            ),
            "feature5": pd.Series(
                [False, False, False, False, False, False, False, False]
            ),
        }
        self.assertEqual(outliers.keys(), target_outliers.keys())
        for key, value in outliers.items():
            self.assertTrue(value.equals(target_outliers[key]))

        corrector = OutlierCorrector(detection_method="iqr", detection_threshold=10000)
        outliers = corrector._identify_outliers(X)
        target_outliers = {
            "feature1": pd.Series(
                [False, False, False, False, False, False, False, True]
            ),
            "feature2": pd.Series(
                [False, False, False, False, False, False, False, True]
            ),
            "feature3": pd.Series(
                [False, False, False, False, False, False, False, False]
            ),
            "feature4": pd.Series(
                [False, False, False, False, False, False, False, False]
            ),
            "feature5": pd.Series(
                [False, False, False, False, False, False, False, False]
            ),
        }
        self.assertEqual(outliers.keys(), target_outliers.keys())
        for key, value in outliers.items():
            self.assertTrue(value.equals(target_outliers[key]))

        corrector = OutlierCorrector(
            detection_method="modified_z_score", detection_threshold=1
        )
        outliers = corrector._identify_outliers(X)
        target_outliers = {
            "feature1": pd.Series(
                [False, False, False, False, False, False, False, True]
            ),
            "feature2": pd.Series(
                [False, False, False, False, False, False, False, False]
            ),
            "feature3": pd.Series(
                [True, False, False, False, False, False, False, True]
            ),
            "feature4": pd.Series(
                [True, False, False, False, False, False, False, True]
            ),
            "feature5": pd.Series(
                [False, False, False, False, False, False, False, False]
            ),
        }
        self.assertEqual(outliers.keys(), target_outliers.keys())
        for key, value in outliers.items():
            self.assertTrue(value.equals(target_outliers[key]))

        corrector = OutlierCorrector(
            detection_method="modified_z_score", detection_threshold=2
        )
        outliers = corrector._identify_outliers(X)
        target_outliers = {
            "feature1": pd.Series(
                [False, False, False, False, False, False, False, True]
            ),
            "feature2": pd.Series(
                [False, False, False, False, False, False, False, False]
            ),
            "feature3": pd.Series(
                [False, False, False, False, False, False, False, True]
            ),
            "feature4": pd.Series(
                [False, False, False, False, False, False, False, False]
            ),
            "feature5": pd.Series(
                [False, False, False, False, False, False, False, False]
            ),
        }
        self.assertEqual(outliers.keys(), target_outliers.keys())
        for key, value in outliers.items():
            self.assertTrue(value.equals(target_outliers[key]))

        corrector = OutlierCorrector(
            detection_method="modified_z_score", detection_threshold=100000
        )
        outliers = corrector._identify_outliers(X)
        target_outliers = {
            "feature1": pd.Series(
                [False, False, False, False, False, False, False, True]
            ),
            "feature2": pd.Series(
                [False, False, False, False, False, False, False, False]
            ),
            "feature3": pd.Series(
                [False, False, False, False, False, False, False, False]
            ),
            "feature4": pd.Series(
                [False, False, False, False, False, False, False, False]
            ),
            "feature5": pd.Series(
                [False, False, False, False, False, False, False, False]
            ),
        }
        self.assertEqual(outliers.keys(), target_outliers.keys())
        for key, value in outliers.items():
            self.assertTrue(value.equals(target_outliers[key]))

    def test_correct_outliers(self):
        """
        Test _correct_outliers method.
        """
        X = pd.DataFrame(
            {
                "feature1": [100, 100, 100, 100, 100, 100, 100, 1],
                "feature2": [100, 100, 100, 100, 100, 100, np.nan, 1],
                "feature3": [1, 2, 3, 4, 5, 6, 7, 10000],
                "feature4": [1, 2, 3, 4, 5, 6, 7, 8],
                "feature5": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1],
                "feature6": ["a", "b", "c", "d", "e", "f", "g", "h"],
            }
        )
        corrector = OutlierCorrector(correction_method="clip", detection_method="iqr")
        with self.assertRaises(TypeError):
            corrector._correct_outliers(feature=None, outliers=None)
        with self.assertRaises(TypeError):
            corrector._correct_outliers(feature="test", outliers=None)
        with self.assertRaises(TypeError):
            corrector._correct_outliers(feature=1, outliers=None)
        with self.assertRaises(TypeError):
            corrector._correct_outliers(X["feature1"], outliers=None)
        with self.assertRaises(TypeError):
            corrector._correct_outliers(X["feature1"], outliers=[])
        with self.assertRaises(ValueError):
            corrector._correct_outliers(X.loc[:, ["feature1"]],)

        corrector.outliers_ = {
            "feature1": pd.Series(
                [False, False, False, False, False, False, False, True]
            )
        }
        corrected_feature = corrector._correct_outliers(
            X.loc[:, ["feature1"]], inplace=False
        )
        self.assertTrue(
            corrected_feature["feature1"].equals(
                pd.Series([100, 100, 100, 100, 100, 100, 100, 100])
            )
        )
        self.assertTrue(
            X["feature1"].equals(pd.Series([100, 100, 100, 100, 100, 100, 100, 1]))
        )

        corrector.outliers_ = {
            "feature2": pd.Series(
                [False, False, False, False, False, False, False, True]
            )
        }
        corrected_feature = corrector._correct_outliers(
            X.loc[:, ["feature2"]], inplace=False
        )
        self.assertTrue(
            corrected_feature["feature2"].equals(
                pd.Series([100, 100, 100, 100, 100, 100, np.nan, 100])
            )
        )

        corrector.outliers_ = {
            "feature3": pd.Series(
                [False, False, False, False, False, False, False, True]
            )
        }
        corrected_feature = corrector._correct_outliers(
            X.loc[:, ["feature3"]], inplace=False
        )
        self.assertTrue(
            corrected_feature["feature3"].equals(
                pd.Series([1, 2, 3, 4, 5, 6, 7, 16.75])
            )
        )

        corrector = OutlierCorrector(
            correction_method="clip", detection_method="z_score", detection_threshold=1
        )
        corrector.outliers_ = {
            "feature1": pd.Series(
                [False, False, False, False, False, False, False, True]
            )
        }
        corrected_feature = corrector._correct_outliers(
            X.loc[:, ["feature1"]], inplace=False
        )
        self.assertTrue(
            corrected_feature["feature1"].equals(
                pd.Series([100, 100, 100, 100, 100, 100, 100, 52.6232143312659])
            )
        )

        corrector.outliers_ = {
            "feature2": pd.Series(
                [False, False, False, False, False, False, False, True]
            )
        }
        corrected_feature = corrector._correct_outliers(
            X.loc[:, ["feature2"]], inplace=False
        )
        self.assertTrue(
            corrected_feature["feature2"].equals(
                pd.Series([100, 100, 100, 100, 100, 100, np.nan, 48.43866002922937])
            )
        )

        corrector.outliers_ = {
            "feature3": pd.Series(
                [False, False, False, False, False, False, False, True]
            )
        }
        corrected_feature = corrector._correct_outliers(
            X.loc[:, ["feature3"]], inplace=False
        )
        self.assertTrue(
            corrected_feature["feature3"].equals(
                pd.Series([1, 2, 3, 4, 5, 6, 7, 4787.620258282109])
            )
        )

        corrector = OutlierCorrector(correction_method="mean")
        corrector.outliers_ = {
            "feature1": pd.Series(
                [False, False, False, False, False, False, False, True]
            )
        }
        corrected_feature = corrector._correct_outliers(
            X.loc[:, ["feature1"]], inplace=False
        )
        self.assertTrue(
            corrected_feature["feature1"].equals(
                pd.Series([100, 100, 100, 100, 100, 100, 100, np.mean(X["feature1"])])
            )
        )

        corrector.outliers_ = {
            "feature2": pd.Series(
                [False, False, False, False, False, False, False, True]
            )
        }
        corrected_feature = corrector._correct_outliers(
            X.loc[:, ["feature2"]], inplace=False
        )
        self.assertTrue(
            corrected_feature["feature2"].equals(
                pd.Series(
                    [100, 100, 100, 100, 100, 100, np.nan, np.mean(X["feature2"])]
                )
            )
        )

        corrector.outliers_ = {
            "feature3": pd.Series(
                [False, False, False, False, False, False, False, True]
            )
        }
        corrected_feature = corrector._correct_outliers(
            X.loc[:, ["feature3"]], inplace=False
        )
        self.assertTrue(
            corrected_feature["feature3"].equals(
                pd.Series([1, 2, 3, 4, 5, 6, 7, np.mean(X["feature3"])])
            )
        )

        corrector = OutlierCorrector(correction_method="median")
        corrector.outliers_ = {
            "feature1": pd.Series(
                [False, False, False, False, False, False, False, True]
            )
        }
        corrected_feature = corrector._correct_outliers(
            X.loc[:, ["feature1"]], inplace=False
        ).astype(float)
        self.assertTrue(
            corrected_feature["feature1"].equals(
                pd.Series([100, 100, 100, 100, 100, 100, 100, np.median(X["feature1"])])
            )
        )

        corrector.outliers_ = {
            "feature2": pd.Series(
                [False, False, False, False, False, False, False, True]
            )
        }
        corrected_feature = corrector._correct_outliers(
            X.loc[:, ["feature2"]], inplace=False
        ).astype(float)
        self.assertTrue(
            corrected_feature["feature2"].equals(
                pd.Series(
                    [100, 100, 100, 100, 100, 100, np.nan, X["feature2"].median()]
                )
            )
        )

        corrector.outliers_ = {
            "feature3": pd.Series(
                [False, False, False, False, False, False, False, True]
            )
        }
        corrected_feature = corrector._correct_outliers(
            X.loc[:, ["feature3"]], inplace=False
        ).astype(float)
        self.assertTrue(
            corrected_feature["feature3"].equals(
                pd.Series([1, 2, 3, 4, 5, 6, 7, X["feature3"].median()])
            )
        )

    def test_fit(self):
        """
        Test fit method.
        """
        X = pd.DataFrame(
            {
                "feature1": [100, 100, 100, 100, 100, 100, 100, 1],
                "feature2": [100, 100, 100, 100, 100, 100, np.nan, 1],
                "feature3": [1, 2, 3, 4, 5, 6, 7, 10000],
                "feature4": [1, 2, 3, 4, 5, 6, 7, 8],
                "feature5": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1],
                "feature6": ["a", "b", "c", "d", "e", "f", "g", "h"],
            }
        )
        data = DataBase(features=X)
        corrector = OutlierCorrector(detection_method="z_score", detection_threshold=2)
        self.assertIsNone(corrector.outliers_)

        with self.assertRaises(TypeError):
            corrector.fit(data=None)
        with self.assertRaises(TypeError):
            corrector.fit(data="test")
        with self.assertRaises(TypeError):
            corrector.fit(data=1)
        with self.assertRaises(TypeError):
            corrector.fit(data=X)
        corrector.fit(DataBase(features=X, labels=pd.Series([0, 1, 0, 1])))
        with self.assertRaises(AssertionError):
            corrector.fit(DataBase(features=pd.DataFrame()))
        with self.assertRaises(ValueError):
            corrector.fit(data=data, layer_in="test")
        with self.assertRaises(TypeError):
            corrector.fit(data=data, layer_in="processed")

        corrector.fit(data)
        target_outliers = {
            "feature1": pd.Series(
                [False, False, False, False, False, False, False, True]
            ),
            "feature2": pd.Series(
                [False, False, False, False, False, False, False, True]
            ),
            "feature3": pd.Series(
                [False, False, False, False, False, False, False, True]
            ),
            "feature4": pd.Series(
                [False, False, False, False, False, False, False, False]
            ),
            "feature5": pd.Series(
                [False, False, False, False, False, False, False, False]
            ),
        }
        self.assertEqual(corrector.outliers_.keys(), target_outliers.keys())
        for key, value in corrector.outliers_.items():
            self.assertTrue(value.equals(target_outliers[key]))

        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
                "feature2": ["a", "b", "c", "d", "e", "f", "g", "h"],
            }
        )
        data = DataBase(features=X)
        corrector.fit(data)
        target_outliers = {
            "feature1": pd.Series(
                [False, False, False, False, False, False, False, False]
            )
        }
        self.assertEqual(corrector.outliers_.keys(), target_outliers.keys())
        for key, value in corrector.outliers_.items():
            self.assertTrue(value.equals(target_outliers[key]))

    def test_transform(self):
        """
        Test transform method.
        """
        X = pd.DataFrame(
            {
                "feature1": [100, 100, 100, 100, 100, 100, 100, 1],
                "feature2": [100, 100, 100, 100, 100, 100, np.nan, 1],
                "feature3": [1, 2, 3, 4, 5, 6, 7, 10000],
                "feature4": [1, 2, 3, 4, 5, 6, 7, 8],
                "feature5": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1],
            }
        )
        data = DataBase(features=X)
        corrector = OutlierCorrector(
            detection_method="z_score", detection_threshold=2, correction_method="mean"
        )
        self.assertIsNone(corrector.outliers_)
        with self.assertRaises(ValueError):
            corrector.transform(data)

        corrector.fit(data)
        self.assertIsInstance(corrector.outliers_, dict)

        with self.assertRaises(ValueError):
            corrector.transform(data, layer_in="processed")

        processed_data = corrector.transform(data, layer_in="raw")
        self.assertIsInstance(processed_data, DataBase)
        raw_feature = processed_data.get_features(layer="raw").astype(float)
        processed_feature = processed_data.get_features(layer="processed").astype(float)
        target_feature = pd.DataFrame(
            {
                "feature1": [100, 100, 100, 100, 100, 100, 100, X["feature1"].mean()],
                "feature2": [
                    100,
                    100,
                    100,
                    100,
                    100,
                    100,
                    np.nan,
                    X["feature2"].mean(),
                ],
                "feature3": [1, 2, 3, 4, 5, 6, 7, X["feature3"].mean()],
                "feature4": [1, 2, 3, 4, 5, 6, 7, 8],
                "feature5": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1],
            }
        ).astype(float)
        self.assertTrue(raw_feature.equals(processed_feature))
        self.assertTrue(processed_feature.equals(target_feature))

        with self.assertRaises(ValueError):
            corrector.transform(
                DataBase(features=pd.DataFrame({"feature1": [1, 2, 3, 4, 5, 6, 7, 8],}))
            )
        with self.assertRaises(TypeError):
            corrector.transform(None)
        with self.assertRaises(TypeError):
            corrector.transform("test")
        with self.assertRaises(TypeError):
            corrector.transform(1)
        with self.assertRaises(TypeError):
            corrector.transform(X)
        corrector.transform(DataBase(features=X, labels=pd.Series([0, 1, 0, 1])))
        with self.assertRaises(ValueError):
            corrector.transform(data, layer_in="test")
        with self.assertRaises(ValueError):
            corrector.transform(data, layer_out="test")

        corrector.transform(data, layer_in="processed")

    def test_fit_transform(self):
        """
        Test fit_transform method.
        """
        X = pd.DataFrame(
            {
                "feature1": [100, 100, 100, 100, 100, 100, 100, 1],
                "feature2": [100, 100, 100, 100, 100, 100, np.nan, 1],
                "feature3": [1, 2, 3, 4, 5, 6, 7, 10000],
                "feature4": [1, 2, 3, 4, 5, 6, 7, 8],
                "feature5": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1],
            }
        )
        data = DataBase(features=X)
        corrector = OutlierCorrector(
            detection_method="z_score", detection_threshold=2, correction_method="mean"
        )
        self.assertIsNone(corrector.outliers_)

        with self.assertRaises(TypeError):
            corrector.fit_transform(None)
        with self.assertRaises(TypeError):
            corrector.fit_transform("test")
        with self.assertRaises(TypeError):
            corrector.fit_transform(1)
        with self.assertRaises(TypeError):
            corrector.fit_transform(X)
        corrector.fit_transform(DataBase(features=X, labels=pd.Series([0, 1, 0, 1])))
        with self.assertRaises(AssertionError):
            corrector.fit_transform(DataBase(features=pd.DataFrame()))

        processed_data = corrector.fit_transform(data, layer_in="raw")
        self.assertIsInstance(processed_data, DataBase)
        raw_feature = processed_data.get_features(layer="raw").astype(float)
        processed_feature = processed_data.get_features(layer="processed").astype(float)
        target_feature = pd.DataFrame(
            {
                "feature1": [100, 100, 100, 100, 100, 100, 100, X["feature1"].mean()],
                "feature2": [
                    100,
                    100,
                    100,
                    100,
                    100,
                    100,
                    np.nan,
                    X["feature2"].mean(),
                ],
                "feature3": [1, 2, 3, 4, 5, 6, 7, X["feature3"].mean()],
                "feature4": [1, 2, 3, 4, 5, 6, 7, 8],
                "feature5": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1],
            }
        ).astype(float)
        self.assertTrue(raw_feature.equals(target_feature))
        self.assertTrue(processed_feature.equals(target_feature))

        X = pd.DataFrame(
            {
                "feature1": [100, 100, 100, 100, 100, 100, 100, 1],
                "feature2": [100, 100, 100, 100, 100, 100, np.nan, 1],
            }
        )
        data = DataBase(features=X)
        processed_data = corrector.fit_transform(data, layer_in="raw")
        self.assertIsInstance(processed_data, DataBase)
        raw_feature = processed_data.get_features(layer="raw").astype(float)
        processed_feature = processed_data.get_features(layer="processed").astype(float)
        target_feature = pd.DataFrame(
            {
                "feature1": [100, 100, 100, 100, 100, 100, 100, X["feature1"].mean()],
                "feature2": [
                    100,
                    100,
                    100,
                    100,
                    100,
                    100,
                    np.nan,
                    X["feature2"].mean(),
                ],
            }
        ).astype(float)
        self.assertTrue(raw_feature.equals(target_feature))
        self.assertTrue(processed_feature.equals(target_feature))

        corrector.fit_transform(data, layer_in="processed")


class TestFeatureScaler(unittest.TestCase):
    """
        Test FeatureScaler class.
        """

    def setUp(self) -> None:
        """
        """
        pass

    def test_init(self):
        """
        Test __init__ method.
        """
        # Test with default values.
        scaler = FeatureScaler()
        self.assertIsInstance(scaler, FeatureScaler)
        self.assertEqual(scaler.scale_method, "minmax")
        self.assertEqual(scaler.scale_range, (0, 1))
        self.assertIsNone(scaler.scaler_)

        # Test with specified values.
        scaler = FeatureScaler(scale_method="standard", feature_range=(-1, 1))
        self.assertIsInstance(scaler, FeatureScaler)
        self.assertEqual(scaler.scale_method, "standard")
        self.assertIsNone(scaler.scale_range)
        self.assertIsNone(scaler.scaler_)

        scaler = FeatureScaler(scale_method="robust")
        self.assertIsInstance(scaler, FeatureScaler)
        self.assertEqual(scaler.scale_method, "robust")
        self.assertIsNone(scaler.scale_range)
        self.assertIsNone(scaler.scaler_)

        with self.assertRaises(ValueError):
            scaler = FeatureScaler(scale_method="test")
        with self.assertRaises(ValueError):
            scaler = FeatureScaler(feature_range=(1, 0))
        with self.assertRaises(ValueError):
            scaler = FeatureScaler(feature_range="test")
        with self.assertRaises(ValueError):
            scaler = FeatureScaler(feature_range=(0, 0))
        with self.assertRaises(ValueError):
            scaler = FeatureScaler(feature_range=(1, 1))
        with self.assertRaises(ValueError):
            scaler = FeatureScaler(feature_range=(1, 2, 3))
        with self.assertRaises(ValueError):
            scaler = FeatureScaler(feature_range=(1,))
        with self.assertRaises(TypeError):
            scaler = FeatureScaler(feature_range=("1", "2"))

    def test__select_scaler(self):
        """
        Test _select_scaler method.
        """
        scaler = FeatureScaler(scale_method="standard")._select_scaler()
        self.assertIsInstance(scaler, StandardScaler)

        scaler = FeatureScaler(
            scale_method="minmax", feature_range=(0, 1)
        )._select_scaler()
        self.assertIsInstance(scaler, MinMaxScaler)
        self.assertEqual(scaler.feature_range, (0, 1))

        scaler = FeatureScaler(
            scale_method="minmax", feature_range=(-1, 1)
        )._select_scaler()
        self.assertIsInstance(scaler, MinMaxScaler)
        self.assertEqual(scaler.feature_range, (-1, 1))

        scaler = FeatureScaler(scale_method="robust")._select_scaler()
        self.assertIsInstance(scaler, RobustScaler)

    def test_fit(self):
        """
        Test fit method.
        """
        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
                "feature2": [1, 2, 3, np.nan, 5, 6, 7, 8],
                "feature3": [-1, -2, -3, -4, -5, -6, -7, -8],
                "feature4": [-1, -2, -3, np.nan, -5, -6, -7, -8],
                "feature5": [-1, -2, -3, 0, 0, 1, 2, 3,],
                "feature6": [-1, -2, -3, np.nan, np.nan, 1, 2, 3,],
                "feature7": [0, 0, 0, 0, 0, 0, 0, 0],
                "feature8": [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                "feature9": [0, 1, 0, 1, 0, 1, 0, 1],
            }
        )
        data = DataBase(features=X)
        scaler = FeatureScaler(scale_method="standard")

        scaler.fit(data=data, layer_in="raw")
        self.assertIsInstance(scaler.scaler_, StandardScaler)
        check_is_fitted(scaler.scaler_)
        assert np.array_equal(scaler.scaler_.mean_, X.mean().values, equal_nan=True)
        assert np.array_equal(scaler.scaler_.var_, X.var(ddof=0).values, equal_nan=True)
        self.assertEqual(scaler.scaler_.n_features_in_, X.shape[1])

        scaler = FeatureScaler(scale_method="minmax", feature_range=(0, 1))
        scaler.fit(data=data, layer_in="raw")
        self.assertIsInstance(scaler.scaler_, MinMaxScaler)
        check_is_fitted(scaler.scaler_)
        self.assertEqual(scaler.scaler_.feature_range, (0, 1))
        assert np.array_equal(scaler.scaler_.data_min_, X.min().values, equal_nan=True)
        assert np.array_equal(scaler.scaler_.data_max_, X.max().values, equal_nan=True)
        self.assertEqual(scaler.scaler_.n_features_in_, X.shape[1])

        scaler = FeatureScaler(scale_method="minmax", feature_range=(-1, 1))
        scaler.fit(data=data, layer_in="raw")
        self.assertIsInstance(scaler.scaler_, MinMaxScaler)
        check_is_fitted(scaler.scaler_)
        self.assertEqual(scaler.scaler_.feature_range, (-1, 1))
        assert np.array_equal(scaler.scaler_.data_min_, X.min().values, equal_nan=True)
        assert np.array_equal(scaler.scaler_.data_max_, X.max().values, equal_nan=True)
        self.assertEqual(scaler.scaler_.n_features_in_, X.shape[1])

        scaler = FeatureScaler(scale_method="robust")
        scaler.fit(data=data, layer_in="raw")
        self.assertIsInstance(scaler.scaler_, RobustScaler)
        check_is_fitted(scaler.scaler_)
        assert np.array_equal(scaler.scaler_.center_, X.median().values, equal_nan=True)
        scale = np.nanpercentile(X, 75, axis=0) - np.nanpercentile(X, 25, axis=0)
        scale[6] = 1.0
        assert np.array_equal(scaler.scaler_.scale_, scale, equal_nan=True,)
        self.assertEqual(scaler.scaler_.n_features_in_, X.shape[1])

        with self.assertRaises(TypeError):
            scaler.fit(data=None)
        with self.assertRaises(TypeError):
            scaler.fit(data="test")
        with self.assertRaises(TypeError):
            scaler.fit(data=1)
        with self.assertRaises(TypeError):
            scaler.fit(data=X)
        scaler.fit(DataBase(features=X, labels=pd.Series([0, 1, 0, 1])))
        with self.assertRaises(AssertionError):
            scaler.fit(DataBase(features=pd.DataFrame()))
        with self.assertRaises(ValueError):
            scaler.fit(data=data, layer_in="test")
        with self.assertRaises(TypeError):
            scaler.fit(data=data, layer_in="processed")

    def test_transform(self):
        """
        Test transform method.
        """
        scaler = FeatureScaler(scale_method="standard")
        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
                "feature2": [1, 2, 3, np.nan, 5, 6, 7, 8],
                "feature3": [-1, -2, -3, -4, -5, -6, -7, -8],
                "feature4": [-1, -2, -3, np.nan, -5, -6, -7, -8],
                "feature5": [0, 0, 0, 0, 0, 0, 0, 0],
                "feature6": [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                "feature7": [0, 1, 0, 1, 0, 1, 0, 1],
            }
        )

        data = DataBase(features=X)
        with self.assertRaises(ValueError):
            scaler.transform(data)

        scaler.fit(data=data, layer_in="raw")

        scaled_data = scaler.transform(data, layer_in="raw")
        self.assertIsInstance(scaled_data, DataBase)
        raw_feature = scaled_data.get_features(layer="raw")
        processed_feature = scaled_data.get_features(layer="processed")
        target_feature = pd.DataFrame(
            {
                "feature1": (X["feature1"] - X["feature1"].mean())
                / X["feature1"].std(axis=0, ddof=0),
                "feature2": (X["feature2"] - X["feature2"].mean())
                / X["feature2"].std(axis=0, ddof=0),
                "feature3": (X["feature3"] - X["feature3"].mean())
                / X["feature3"].std(axis=0, ddof=0),
                "feature4": (X["feature4"] - X["feature4"].mean())
                / X["feature4"].std(axis=0, ddof=0),
                "feature5": X["feature5"],
                "feature6": (X["feature6"] - X["feature6"].mean())
                / X["feature6"].std(axis=0, ddof=0),
                "feature7": (X["feature7"] - X["feature7"].mean())
                / X["feature7"].std(axis=0, ddof=0),
            }
        ).astype(float)
        self.assertTrue(raw_feature.equals(X))
        self.assertTrue(processed_feature.equals(target_feature))

        scaler = FeatureScaler(scale_method="minmax", feature_range=(0, 1))
        scaler.fit(data=data, layer_in="raw")
        scaled_data = scaler.transform(data, layer_in="raw")
        self.assertIsInstance(scaled_data, DataBase)
        raw_feature = scaled_data.get_features(layer="raw")
        processed_feature = scaled_data.get_features(layer="processed")
        scale = 1 / (X.max(axis=0) - X.min(axis=0))
        target_feature = scale * X - X.min(axis=0) * scale
        target_feature["feature5"] = 0.0
        self.assertTrue(raw_feature.equals(X))
        self.assertTrue(processed_feature.equals(target_feature))

        scaler = FeatureScaler(scale_method="minmax", feature_range=(-1, 1))
        scaler.fit(data=data, layer_in="raw")
        scaled_data = scaler.transform(data, layer_in="raw")
        self.assertIsInstance(scaled_data, DataBase)
        raw_feature = scaled_data.get_features(layer="raw")
        processed_feature = scaled_data.get_features(layer="processed")
        scale = 2 / (X.max(axis=0) - X.min(axis=0))
        min_ = -1 - X.min(axis=0) * scale
        target_feature = scale * X + min_
        target_feature["feature5"] = -1.0
        self.assertTrue(raw_feature.equals(X))
        self.assertTrue(processed_feature.equals(target_feature))

        scaler = FeatureScaler(scale_method="robust")
        scaler.fit(data=data, layer_in="raw")
        scaled_data = scaler.transform(data, layer_in="raw")
        self.assertIsInstance(scaled_data, DataBase)
        raw_feature = scaled_data.get_features(layer="raw")
        processed_feature = scaled_data.get_features(layer="processed")
        scale = np.nanpercentile(X, 75, axis=0) - np.nanpercentile(X, 25, axis=0)
        scale[6] = 1.0
        center = np.nanmedian(X, axis=0)
        target_feature = (X - center) / scale
        target_feature["feature5"] = 0.0
        self.assertTrue(raw_feature.equals(X))
        self.assertTrue(processed_feature.equals(target_feature))

        with self.assertRaises(ValueError):
            scaler.transform(
                DataBase(
                    features=pd.DataFrame({"feature1": [1, 2, 3, 4, 5, 6, 7, 8],})
                ),
                layer_in="raw",
            )
        with self.assertRaises(TypeError):
            scaler.transform(X, layer_in="raw")
        with self.assertRaises(TypeError):
            scaler.transform(X, layer_in="test")

        scaler.transform(data, layer_in="processed")

    def test_fit_transform(self):
        """
        Test fit_transform method.
        """
        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
                "feature2": [1, 2, 3, np.nan, 5, 6, 7, 8],
                "feature3": [-1, -2, -3, -4, -5, -6, -7, -8],
                "feature4": [-1, -2, -3, np.nan, -5, -6, -7, -8],
                "feature5": [0, 0, 0, 0, 0, 0, 0, 0],
                "feature6": [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                "feature7": [0, 1, 0, 1, 0, 1, 0, 1],
            }
        )

        data = DataBase(features=X)
        scaler = FeatureScaler(scale_method="standard")

        with self.assertRaises(TypeError):
            scaler.fit_transform(None)
        with self.assertRaises(ValueError):
            scaler.fit_transform(data, layer_in="test")
        with self.assertRaises(TypeError):
            scaler.fit_transform(data, layer_in="processed")

        scaled_data = scaler.fit_transform(data, layer_in="raw")
        self.assertIsInstance(scaled_data, DataBase)
        raw_feature = scaled_data.get_features(layer="raw")
        processed_feature = scaled_data.get_features(layer="processed")
        target_feature = pd.DataFrame(
            {
                "feature1": (X["feature1"] - X["feature1"].mean())
                / X["feature1"].std(axis=0, ddof=0),
                "feature2": (X["feature2"] - X["feature2"].mean())
                / X["feature2"].std(axis=0, ddof=0),
                "feature3": (X["feature3"] - X["feature3"].mean())
                / X["feature3"].std(axis=0, ddof=0),
                "feature4": (X["feature4"] - X["feature4"].mean())
                / X["feature4"].std(axis=0, ddof=0),
                "feature5": X["feature5"],
                "feature6": (X["feature6"] - X["feature6"].mean())
                / X["feature6"].std(axis=0, ddof=0),
                "feature7": (X["feature7"] - X["feature7"].mean())
                / X["feature7"].std(axis=0, ddof=0),
            }
        ).astype(float)
        self.assertTrue(raw_feature.equals(X))
        self.assertTrue(processed_feature.equals(target_feature))

        scaler.fit_transform(data, layer_in="processed")


class TestCorrelatedFeatureFilter(unittest.TestCase):
    """
    Test CorrelatedFeatureFilter class.
    """

    def setUp(self) -> None:
        """
        """
        pass

    def test_init(self):
        """
        Test __init__ method.
        """
        # Test with default values.
        filter = CorrelatedFeatureFilter()
        self.assertIsInstance(filter, CorrelatedFeatureFilter)
        self.assertEqual(filter.feature_corr_threshold, 0.9)
        self.assertIsNone(filter.dropped_features_)

        # Test with specified values.
        filter = CorrelatedFeatureFilter(feature_corr_threshold=0)
        self.assertIsInstance(filter, CorrelatedFeatureFilter)
        self.assertEqual(filter.feature_corr_threshold, 0)
        self.assertIsNone(filter.dropped_features_)

        filter = CorrelatedFeatureFilter(feature_corr_threshold=0.5)
        self.assertIsInstance(filter, CorrelatedFeatureFilter)
        self.assertEqual(filter.feature_corr_threshold, 0.5)
        self.assertIsNone(filter.dropped_features_)

        filter = CorrelatedFeatureFilter(feature_corr_threshold=1)
        self.assertIsInstance(filter, CorrelatedFeatureFilter)
        self.assertEqual(filter.feature_corr_threshold, 1)
        self.assertIsNone(filter.dropped_features_)

        filter = CorrelatedFeatureFilter(feature_corr_threshold=None)
        with self.assertRaises(TypeError):
            filter = CorrelatedFeatureFilter(feature_corr_threshold="test")
        with self.assertRaises(ValueError):
            filter = CorrelatedFeatureFilter(feature_corr_threshold=-0.1)
        with self.assertRaises(ValueError):
            filter = CorrelatedFeatureFilter(feature_corr_threshold=1.1)

    def test_identify_highly_correlated_features(self):
        """
        Test _identify_highly_correlated_features method.
        """
        X = pd.DataFrame(
            {
                "feature1": [0, 1, 0, 1, 0],
                "feature2": [1, 0, 1, 0, 1],
                "feature3": [0, 0, 1, 1, 0],
                "feature4": [1.0, 2.0, 3.0, 4.0, np.nan],
                "feature5": [0.1, 10, np.nan, 5, 0.3],
                "feature6": [-10, 0.1, np.nan, 0.2, -20],
                "feature7": [np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])
        filter = CorrelatedFeatureFilter(feature_corr_threshold=0.9)
        dropped_features = filter._identify_highly_correlated_features(X, y)
        dropped_features.sort()
        self.assertListEqual(dropped_features, ["feature2", "feature6"])

        filter = CorrelatedFeatureFilter(feature_corr_threshold=0)
        dropped_features = filter._identify_highly_correlated_features(X, y)
        dropped_features.sort()
        self.assertListEqual(
            dropped_features,
            ["feature2", "feature3", "feature4", "feature5", "feature6"],
        )

        filter = CorrelatedFeatureFilter(feature_corr_threshold=1)
        dropped_features = filter._identify_highly_correlated_features(X, y)
        dropped_features.sort()
        self.assertListEqual(dropped_features, ["feature2"])

        X = pd.DataFrame(
            {
                "feature1": [0, 1, 0, 1, 0],
                "feature2": [0, 1, np.nan, 1, 0],
                "feature3": [1, 0, 1, 0, 1],
                "feature4": [0, 0, 1, 1, 0],
                "feature5": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature6": [0.1, 10, 0.2, 5, 0.3],
                "feature7": [-10, 0.1, -5, 0.2, -20],
                "feature8": [np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )
        y = pd.Series([0.1, 10, 0.2, 5, 0.3])

        filter = CorrelatedFeatureFilter(feature_corr_threshold=0.9)
        dropped_features = filter._identify_highly_correlated_features(X, y)
        dropped_features.sort()
        self.assertListEqual(
            dropped_features, ["feature1", "feature2", "feature3", "feature7"]
        )

        filter = CorrelatedFeatureFilter(feature_corr_threshold=0)
        dropped_features = filter._identify_highly_correlated_features(X, y)
        dropped_features.sort()
        self.assertListEqual(
            dropped_features,
            ["feature1", "feature2", "feature3", "feature4", "feature5", "feature7"],
        )

        filter = CorrelatedFeatureFilter(feature_corr_threshold=1)
        dropped_features = filter._identify_highly_correlated_features(X, y)
        dropped_features.sort()
        self.assertListEqual(dropped_features, ["feature2", "feature3"])

        with self.assertRaises(TypeError):
            filter._identify_highly_correlated_features(None, y)
        with self.assertRaises(TypeError):
            filter._identify_highly_correlated_features(X, None)
        with self.assertRaises(TypeError):
            filter._identify_highly_correlated_features(1, y)
        with self.assertRaises(ValueError):
            filter._identify_highly_correlated_features(X, pd.Series([0.1, 10, 0.2, 5]))

    def test_fit(self):
        """
        Test fit method.
        """
        X = pd.DataFrame(
            {
                "feature1": [0, 1, 0, 1, 0],
                "feature2": [1, 0, 1, 0, 1],
                "feature3": [0, 0, 1, 1, 0],
                "feature4": [1.0, 2.0, 3.0, 4.0, np.nan],
                "feature5": [0.1, 10, np.nan, 5, 0.3],
                "feature6": [-10, 0.1, np.nan, 0.2, -20],
                "feature7": [np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])
        data = DataBase(features=X, labels=y)
        filter = CorrelatedFeatureFilter()

        self.assertIsNone(filter.dropped_features_)

        filter.fit(data, layer_in="raw")
        self.assertIsInstance(filter.dropped_features_, list)
        filter.dropped_features_.sort()
        self.assertListEqual(filter.dropped_features_, ["feature2", "feature6"])

        with self.assertRaises(TypeError):
            filter.fit(X, layer_in="raw")
        with self.assertRaises(ValueError):
            filter.fit(data, layer_in="test")
        with self.assertRaises(TypeError):
            filter.fit(1, layer_in="raw")
        with self.assertRaises(TypeError):
            filter.fit(data, layer_in="processed")
        with self.assertRaises(TypeError):
            filter.fit(DataBase(features=X))
        filter.fit(DataBase(features=X, labels=pd.Series([0, 1, 0, 1])))

        filter.fit(data, layer_in="raw")

    def test_transform(self):
        """
        Test transform method.
        """
        X = pd.DataFrame(
            {
                "feature1": [0, 1, 0, 1, 0],
                "feature2": [1, 0, 1, 0, 1],
                "feature3": [0, 0, 1, 1, 0],
                "feature4": [1.0, 2.0, 3.0, 4.0, np.nan],
                "feature5": [0.1, 10, np.nan, 5, 0.3],
                "feature6": [-10, 0.1, np.nan, 0.2, -20],
                "feature7": [np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])
        data = DataBase(features=X, labels=y)
        filter = CorrelatedFeatureFilter()

        with self.assertRaises(ValueError):
            filter.transform(data, layer_in="raw")

        filter.fit(data, layer_in="raw")

        with self.assertRaises(ValueError):
            filter.transform(data, layer_in="processed")

        transformed_data = filter.transform(data, layer_in="raw", layer_out="processed")
        self.assertIsInstance(transformed_data, DataBase)
        raw_feature = transformed_data.get_features(layer="raw")
        processed_feature = transformed_data.get_features(layer="processed")
        target_feature = X.drop(columns=filter.dropped_features_)
        self.assertTrue(raw_feature.equals(X))
        self.assertTrue(processed_feature.equals(target_feature))

        with self.assertRaises(TypeError):
            filter.transform(X, layer_in="raw", layer_out="processed")
        with self.assertRaises(TypeError):
            filter.transform(1, layer_in="raw", layer_out="processed")
        with self.assertRaises(ValueError):
            filter.transform(data, layer_in="test", layer_out="processed")
        with self.assertRaises(ValueError):
            filter.transform(data, layer_in="raw", layer_out="test")
        with self.assertRaises(ValueError):
            filter.transform(DataBase(features=X.loc[:, ["feature1"]]))

        with self.assertRaises(ValueError):
            filter.transform(data, layer_in="processed", layer_out="processed")

    def test_fit_transform(self):
        """
        Test fit_transform method.
        """
        X = pd.DataFrame(
            {
                "feature1": [0, 1, 0, 1, 0],
                "feature2": [1, 0, 1, 0, 1],
                "feature3": [0, 0, 1, 1, 0],
                "feature4": [1.0, 2.0, 3.0, 4.0, np.nan],
                "feature5": [0.1, 10, np.nan, 5, 0.3],
                "feature6": [-10, 0.1, np.nan, 0.2, -20],
                "feature7": [np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])
        data = DataBase(features=X, labels=y)
        filter = CorrelatedFeatureFilter()

        with self.assertRaises(TypeError):
            filter.fit_transform(None)
        with self.assertRaises(TypeError):
            filter.fit_transform(X)
        with self.assertRaises(ValueError):
            filter.fit_transform(data, layer_in="test")
        with self.assertRaises(TypeError):
            filter.fit_transform(data, layer_in="processed")

        transformed_data = filter.fit_transform(data, layer_in="raw")
        self.assertIsInstance(transformed_data, DataBase)
        raw_feature = transformed_data.get_features(layer="raw")
        processed_feature = transformed_data.get_features(layer="processed")
        target_feature = X.drop(columns=filter.dropped_features_)
        self.assertTrue(raw_feature.equals(X))
        self.assertTrue(processed_feature.equals(target_feature))

        filter.fit_transform(data, layer_in="processed")


class TestDataProcessorBase(unittest.TestCase):
    """
    Test DataProcessorBase class.
    """

    def test_init(self):
        processor = DataProcessorBase()
        self.assertIsInstance(processor, DataProcessorBase)
        self.assertEqual(processor.null_ratio_threshold, 0.5)
        self.assertEqual(processor.impute_strategy, "mean")
        self.assertIsNone(processor.impute_fill_value)

        self.assertIsInstance(
            processor.high_quality_feature_filter, HighQualityFeatureFilter
        )
        self.assertIsInstance(processor.missing_value_imputer, MissingValueImputer)
        self.assertIsInstance(processor.processors[0], HighQualityFeatureFilter)
        self.assertIsInstance(processor.processors[1], MissingValueImputer)
        self.assertEqual(
            processor.high_quality_feature_filter.null_ratio_threshold, 0.5
        )
        self.assertIsNone(processor.high_quality_feature_filter.dominance_threshold)
        self.assertIsNone(processor.high_quality_feature_filter.variance_threshold)
        self.assertIsNone(processor.high_quality_feature_filter.selected_features_)
        self.assertEqual(processor.missing_value_imputer.impute_strategy, "mean")
        self.assertIsNone(processor.missing_value_imputer.impute_fill_value)
        self.assertIsNone(processor.missing_value_imputer.impute_values_)

        processor = DataProcessorBase(
            null_ratio_threshold=0,
            impute_strategy="most_frequent",
            impute_fill_value=None,
        )
        self.assertIsInstance(processor, DataProcessorBase)
        self.assertEqual(processor.null_ratio_threshold, 0)
        self.assertEqual(processor.impute_strategy, "most_frequent")
        self.assertIsNone(processor.impute_fill_value)
        self.assertIsInstance(processor.processors[0], HighQualityFeatureFilter)
        self.assertIsInstance(processor.processors[1], MissingValueImputer)
        self.assertEqual(processor.high_quality_feature_filter.null_ratio_threshold, 0)
        self.assertIsNone(processor.high_quality_feature_filter.dominance_threshold)
        self.assertIsNone(processor.high_quality_feature_filter.variance_threshold)
        self.assertIsNone(processor.high_quality_feature_filter.selected_features_)
        self.assertEqual(
            processor.missing_value_imputer.impute_strategy, "most_frequent"
        )
        self.assertIsNone(processor.missing_value_imputer.impute_fill_value)
        self.assertIsNone(processor.missing_value_imputer.impute_values_)

        processor = DataProcessorBase(
            null_ratio_threshold=0.9, impute_strategy="constant", impute_fill_value=0,
        )
        self.assertIsInstance(processor, DataProcessorBase)
        self.assertEqual(processor.null_ratio_threshold, 0.9)
        self.assertEqual(processor.impute_strategy, "constant")
        self.assertEqual(processor.impute_fill_value, 0)

        self.assertEqual(
            processor.high_quality_feature_filter.null_ratio_threshold, 0.9
        )
        self.assertIsNone(processor.high_quality_feature_filter.dominance_threshold)
        self.assertIsNone(processor.high_quality_feature_filter.variance_threshold)
        self.assertIsNone(processor.high_quality_feature_filter.selected_features_)
        self.assertEqual(processor.missing_value_imputer.impute_strategy, "constant")
        self.assertEqual(processor.missing_value_imputer.impute_fill_value, 0)
        self.assertIsNone(processor.missing_value_imputer.impute_values_)

        with self.assertRaises(TypeError):
            processor = DataProcessorBase(null_ratio_threshold="1")
        with self.assertRaises(ValueError):
            processor = DataProcessorBase(null_ratio_threshold=-0.1)
        with self.assertRaises(ValueError):
            processor = DataProcessorBase(null_ratio_threshold=1)
        with self.assertRaises(ValueError):
            processor = DataProcessorBase(impute_strategy="test")
        with self.assertRaises(TypeError):
            processor = DataProcessorBase(impute_fill_value="test")

    def test_fit(self):
        """
        Test fit method.
        """
        feature_matrix = pd.DataFrame(
            {
                "feature1": [1, np.nan, np.nan, 2],
                "feature2": [1, np.nan, np.nan, np.nan],
                "feature3": [1, 1, 0, 0],
                "feature4": [1, 1, 1, 0],
                "feature5": [1, 1.001, 1.002, 1.003],
                "feature6": [1.5, 2.5, 3.6, 4.8],
            }
        )
        data = DataBase(features=feature_matrix)

        processor = DataProcessorBase(
            null_ratio_threshold=0.5, impute_strategy="mean", impute_fill_value=None,
        )
        processor.fit(data)

        self.assertIsInstance(
            processor.high_quality_feature_filter.selected_features_, pd.Index
        )
        self.assertListEqual(
            processor.high_quality_feature_filter.selected_features_.tolist(),
            ["feature1", "feature3", "feature4", "feature5", "feature6"],
        )
        self.assertEqual(
            processor.missing_value_imputer.impute_values_,
            {
                "feature1": feature_matrix["feature1"].mean(),
                "feature3": feature_matrix["feature3"].mean(),
                "feature4": feature_matrix["feature4"].mean(),
                "feature5": feature_matrix["feature5"].mean(),
                "feature6": feature_matrix["feature6"].mean(),
            },
        )
        processed_data = data.get_features(layer="processed")
        self.assertTrue(
            processed_data.equals(
                pd.DataFrame(
                    {
                        "feature1": [1, 1.5, 1.5, 2],
                        "feature3": [1, 1, 0, 0],
                        "feature4": [1, 1, 1, 0],
                        "feature5": [1, 1.001, 1.002, 1.003],
                        "feature6": [1.5, 2.5, 3.6, 4.8],
                    }
                )
            )
        )

        processor = DataProcessorBase(
            null_ratio_threshold=0.9,
            impute_strategy="most_frequent",
            impute_fill_value=None,
        )
        processor.fit(data)
        self.assertIsInstance(
            processor.high_quality_feature_filter.selected_features_, pd.Index
        )
        self.assertListEqual(
            processor.high_quality_feature_filter.selected_features_.tolist(),
            ["feature1", "feature2", "feature3", "feature4", "feature5", "feature6"],
        )
        self.assertEqual(
            processor.missing_value_imputer.impute_values_,
            {
                "feature1": 1,
                "feature2": 1,
                "feature3": 0,
                "feature4": 1,
                "feature5": 1,
                "feature6": 1.5,
            },
        )
        processed_data = data.get_features(layer="processed").astype(float)
        self.assertTrue(
            processed_data.equals(
                pd.DataFrame(
                    {
                        "feature1": [1, 1, 1, 2],
                        "feature2": [1, 1, 1, 1],
                        "feature3": [1, 1, 0, 0],
                        "feature4": [1, 1, 1, 0],
                        "feature5": [1, 1.001, 1.002, 1.003],
                        "feature6": [1.5, 2.5, 3.6, 4.8],
                    }
                ).astype(float)
            )
        )

        processor = DataProcessorBase(
            null_ratio_threshold=0, impute_strategy="constant", impute_fill_value=0,
        )
        processor.fit(data)
        self.assertIsInstance(
            processor.high_quality_feature_filter.selected_features_, pd.Index
        )
        self.assertListEqual(
            processor.high_quality_feature_filter.selected_features_.tolist(),
            ["feature3", "feature4", "feature5", "feature6"],
        )
        self.assertEqual(
            processor.missing_value_imputer.impute_values_,
            {"feature3": 0, "feature4": 0, "feature5": 0, "feature6": 0,},
        )
        processed_data = data.get_features(layer="processed")
        self.assertTrue(
            processed_data.equals(
                feature_matrix[["feature3", "feature4", "feature5", "feature6"]]
            )
        )

        with self.assertRaises(TypeError):
            processor.fit(None)
        with self.assertRaises(TypeError):
            processor.fit(feature_matrix)
        with self.assertRaises(AssertionError):
            processor.fit(DataBase(pd.DataFrame()))

    def test_transform(self):
        """
        Test transform method.
        """
        feature_matrix = pd.DataFrame(
            {
                "feature1": [1, np.nan, np.nan, 2],
                "feature2": [1, np.nan, np.nan, np.nan],
                "feature3": [1, 1, 0, 0],
                "feature4": [1, 1, 1, 0],
                "feature5": [1, 1.001, 1.002, 1.003],
                "feature6": [1.5, 2.5, 3.6, 4.8],
            }
        )
        data = DataBase(features=feature_matrix)
        processor = DataProcessorBase(
            null_ratio_threshold=0.5, impute_strategy="mean", impute_fill_value=None,
        )
        with self.assertRaises(ValueError):
            processor.transform(data)

        processor.fit(data)
        transformed_data = processor.transform(data)
        self.assertIsInstance(transformed_data, DataBase)
        raw_feature = transformed_data.get_features(layer="raw")
        processed_feature = transformed_data.get_features(layer="processed")
        self.assertTrue(raw_feature.equals(feature_matrix))
        self.assertTrue(
            processed_feature.equals(
                pd.DataFrame(
                    {
                        "feature1": [1, 1.5, 1.5, 2],
                        "feature3": [1, 1, 0, 0],
                        "feature4": [1, 1, 1, 0],
                        "feature5": [1, 1.001, 1.002, 1.003],
                        "feature6": [1.5, 2.5, 3.6, 4.8],
                    }
                )
            )
        )
        with self.assertRaises(ValueError):
            processor.transform(
                DataBase(features=feature_matrix.loc[:, ["feature1", "feature2"]])
            )

        feature_matrix = pd.DataFrame(
            {
                "feature1": [1, np.nan, np.nan, 2],
                "feature2": [1, np.nan, np.nan, np.nan],
                "feature3": [1, 1, 0, 0],
                "feature4": [1, 1, 1, 0],
                "feature5": [1, 1.001, 1.002, 1.003],
                "feature6": [1.5, 2.5, 3.6, 4.8],
                "feature7": [np.nan, np.nan, np.nan, np.nan],
            }
        )
        data = DataBase(features=feature_matrix)
        transformed_data = processor.transform(data)
        self.assertIsInstance(transformed_data, DataBase)
        raw_feature = transformed_data.get_features(layer="raw")
        processed_feature = transformed_data.get_features(layer="processed")
        self.assertTrue(raw_feature.equals(feature_matrix))
        self.assertTrue(
            processed_feature.equals(
                pd.DataFrame(
                    {
                        "feature1": [1, 1.5, 1.5, 2],
                        "feature3": [1, 1, 0, 0],
                        "feature4": [1, 1, 1, 0],
                        "feature5": [1, 1.001, 1.002, 1.003],
                        "feature6": [1.5, 2.5, 3.6, 4.8],
                    }
                )
            )
        )

        with self.assertRaises(TypeError):
            processor.transform(feature_matrix)

    def test_fit_transform(self):
        """
        Test fit_transform method.
        """
        feature_matrix = pd.DataFrame(
            {
                "feature1": [1, np.nan, np.nan, 2],
                "feature2": [1, np.nan, np.nan, np.nan],
                "feature3": [1, 1, 0, 0],
                "feature4": [1, 1, 1, 0],
                "feature5": [1, 1.001, 1.002, 1.003],
                "feature6": [1.5, 2.5, 3.6, 4.8],
            }
        )
        data = DataBase(features=feature_matrix)
        processor = DataProcessorBase(
            null_ratio_threshold=0.5,
            impute_strategy="most_frequent",
            impute_fill_value=None,
        )
        transformed_data = processor.fit_transform(data)
        self.assertIsInstance(transformed_data, DataBase)
        raw_feature = transformed_data.get_features(layer="raw")
        processed_feature = transformed_data.get_features(layer="processed").astype(
            float
        )
        self.assertTrue(raw_feature.equals(feature_matrix))
        self.assertTrue(
            processed_feature.equals(
                pd.DataFrame(
                    {
                        "feature1": [1, 1, 1, 2],
                        "feature3": [1, 1, 0, 0],
                        "feature4": [1, 1, 1, 0],
                        "feature5": [1, 1.001, 1.002, 1.003],
                        "feature6": [1.5, 2.5, 3.6, 4.8],
                    }
                ).astype(float)
            )
        )

        with self.assertRaises(TypeError):
            processor.fit_transform(None)
        with self.assertRaises(TypeError):
            processor.fit_transform(feature_matrix)


class TestBinaryDataProcessor(unittest.TestCase):
    """
    Test BinaryDataProcessor class.
    """

    def test_init(self):
        """
        Test __init__ method.
        """
        processor = BinaryDataProcessor()
        self.assertIsInstance(processor, BinaryDataProcessor)
        self.assertIsInstance(processor, DataProcessorBase)
        self.assertEqual(processor.null_ratio_threshold, 0.5)
        self.assertEqual(processor.dominance_threshold, 0.95)
        self.assertEqual(processor.impute_strategy, "most_frequent")
        self.assertEqual(processor.impute_fill_value, 0)
        self.assertEqual(processor.chi2_p_threshold, 0.5)

        self.assertIsInstance(processor.processors[0], HighQualityFeatureFilter)
        self.assertIsInstance(processor.processors[1], InformativeFeatureFilter)
        self.assertIsInstance(processor.processors[2], MissingValueImputer)
        self.assertEqual(processor.processors[0].dominance_threshold, 0.95)
        self.assertIsNone(processor.processors[1].target_corr_threshold)
        self.assertEqual(processor.processors[1].chi2_p_threshold, 0.5)
        self.assertIsNone(processor.processors[1].fold_change_threshold)

        processor = BinaryDataProcessor(
            null_ratio_threshold=0,
            dominance_threshold=0.5,
            chi2_p_threshold=0.1,
            impute_strategy="constant",
            impute_fill_value=1,
        )
        self.assertIsInstance(processor, BinaryDataProcessor)
        self.assertIsInstance(processor, DataProcessorBase)
        self.assertEqual(processor.null_ratio_threshold, 0)
        self.assertEqual(processor.dominance_threshold, 0.5)
        self.assertEqual(processor.impute_strategy, "constant")
        self.assertEqual(processor.impute_fill_value, 1)
        self.assertEqual(processor.chi2_p_threshold, 0.1)

        self.assertIsInstance(processor.processors[0], HighQualityFeatureFilter)
        self.assertIsInstance(processor.processors[1], InformativeFeatureFilter)
        self.assertIsInstance(processor.processors[2], MissingValueImputer)
        self.assertEqual(processor.processors[0].null_ratio_threshold, 0)
        self.assertEqual(processor.processors[0].dominance_threshold, 0.5)
        self.assertIsNone(processor.processors[1].target_corr_threshold)
        self.assertEqual(processor.processors[1].chi2_p_threshold, 0.1)
        self.assertIsNone(processor.processors[1].fold_change_threshold)
        self.assertEqual(processor.processors[2].impute_strategy, "constant")
        self.assertEqual(processor.processors[2].impute_fill_value, 1)

        with self.assertRaises(TypeError):
            processor = BinaryDataProcessor(dominance_threshold="1")
        with self.assertRaises(ValueError):
            processor = BinaryDataProcessor(dominance_threshold=0.4)
        with self.assertRaises(ValueError):
            processor = BinaryDataProcessor(dominance_threshold=1.1)

        with self.assertRaises(TypeError):
            processor = BinaryDataProcessor(chi2_p_threshold="1")
        with self.assertRaises(ValueError):
            processor = BinaryDataProcessor(dominance_threshold=-1)
        with self.assertRaises(ValueError):
            processor = BinaryDataProcessor(dominance_threshold=1.1)

        with self.assertRaises(ValueError):
            processor = BinaryDataProcessor(impute_strategy="mean")
        with self.assertRaises(ValueError):
            processor = BinaryDataProcessor(impute_strategy="median")
        with self.assertRaises(ValueError):
            processor = BinaryDataProcessor(impute_strategy="test")

    def test_fit(self):
        """
        Test fit method.
        """
        X = pd.DataFrame(
            {
                "feature1": [0, 1, 0, 1, 0, 1],
                "feature2": [1, 0, 1, 0, 1, np.nan],
                "feature3": [0, 0, 1, 1, 0, np.nan],
                "feature4": [1, np.nan, np.nan, np.nan, 1, 1],
                "feature5": [1, 1, 1, 1, 1, 0],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0, 1])
        data = DataBase(features=X, labels=y)
        processor = BinaryDataProcessor(
            null_ratio_threshold=0.5,
            dominance_threshold=0.7,
            chi2_p_threshold=0.5,
            impute_strategy="constant",
            impute_fill_value=0,
        )
        processor.fit(data)

        self.assertIsInstance(
            processor.high_quality_feature_filter.selected_features_, pd.Index
        )
        self.assertListEqual(
            processor.high_quality_feature_filter.selected_features_.tolist(),
            ["feature1", "feature2", "feature3"],
        )
        self.assertIsInstance(
            processor.informative_feature_filter.identified_features_, pd.Index
        )
        self.assertListEqual(
            processor.informative_feature_filter.identified_features_.tolist(),
            ["feature1", "feature3"],
        )
        self.assertEqual(
            processor.missing_value_imputer.impute_values_,
            {"feature1": 0, "feature3": 0,},
        )

        processor = BinaryDataProcessor(
            null_ratio_threshold=0.5,
            dominance_threshold=1,
            chi2_p_threshold=1,
            impute_strategy="constant",
            impute_fill_value=0,
        )
        processor.fit(data)
        self.assertListEqual(
            processor.high_quality_feature_filter.selected_features_.tolist(),
            ["feature1", "feature2", "feature3", "feature4", "feature5"],
        )
        self.assertListEqual(
            processor.informative_feature_filter.identified_features_.tolist(),
            ["feature1", "feature2", "feature3", "feature4", "feature5"],
        )
        self.assertEqual(
            processor.missing_value_imputer.impute_values_,
            {"feature1": 0, "feature2": 0, "feature3": 0, "feature4": 0, "feature5": 0},
        )

        processor = BinaryDataProcessor(
            null_ratio_threshold=0,
            dominance_threshold=0.6,
            chi2_p_threshold=0.0001,
            impute_strategy="constant",
            impute_fill_value=0,
        )
        with self.assertRaises(RuntimeError):
            processor.fit(data)

        with self.assertRaises(TypeError):
            processor.fit(None)
        with self.assertRaises(TypeError):
            processor.fit(X)

    def test_transform(self):
        """
        Test transform method.
        """
        X = pd.DataFrame(
            {
                "feature1": [0, 1, 0, 1, 0, 1],
                "feature2": [1, 0, 1, 0, 1, np.nan],
                "feature3": [0, 0, 1, 1, 0, np.nan],
                "feature4": [1, np.nan, np.nan, np.nan, 1, 1],
                "feature5": [1, 1, 1, 1, 1, 0],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0, 1])
        data = DataBase(features=X, labels=y)
        processor = BinaryDataProcessor(
            null_ratio_threshold=0.5,
            dominance_threshold=0.7,
            chi2_p_threshold=0.5,
            impute_strategy="constant",
            impute_fill_value=0,
        )
        with self.assertRaises(ValueError):
            processor.transform(data)
        processor.fit(data)
        transformed_data = processor.transform(data)
        self.assertIsInstance(transformed_data, DataBase)
        raw_feature = transformed_data.get_features(layer="raw")
        processed_feature = transformed_data.get_features(layer="processed").astype(
            float
        )
        self.assertTrue(raw_feature.equals(X))
        scaler = StandardScaler()
        target_feature = scaler.fit_transform(
            pd.DataFrame(
                {"feature1": [0, 1, 0, 1, 0, 1], "feature3": [0, 0, 1, 1, 0, 0], }
            )
        )
        target_feature = pd.DataFrame(target_feature, columns=["feature1", "feature3"])
        self.assertTrue(processed_feature.equals(target_feature))

        with self.assertRaises(ValueError):
            processor.transform(DataBase(features=X.loc[:, ["feature1", "feature2"]]))
        with self.assertRaises(TypeError):
            processor.transform(X)

    def test_fit_transform(self):
        """
        Test fit_transform method.
        """
        X = pd.DataFrame(
            {
                "feature1": [0, 1, 0, 1, 0, 1],
                "feature2": [1, 0, 1, 0, 1, np.nan],
                "feature3": [0, 0, 1, 1, 0, np.nan],
                "feature4": [1, np.nan, np.nan, np.nan, 1, 1],
                "feature5": [1, 1, 1, 1, 1, 0],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0, 1])
        data = DataBase(features=X, labels=y)
        processor = BinaryDataProcessor(
            null_ratio_threshold=0.5,
            dominance_threshold=0.7,
            chi2_p_threshold=0.5,
            impute_strategy="constant",
            impute_fill_value=0,
        )
        transformed_data = processor.fit_transform(data)
        self.assertIsInstance(transformed_data, DataBase)
        raw_feature = transformed_data.get_features(layer="raw")
        processed_feature = transformed_data.get_features(layer="processed").astype(
            float
        )
        self.assertTrue(raw_feature.equals(X))
        scaler = StandardScaler()
        target_feature = scaler.fit_transform(
            pd.DataFrame(
                {"feature1": [0, 1, 0, 1, 0, 1], "feature3": [0, 0, 1, 1, 0, 0],}
            )
        )
        target_feature = pd.DataFrame(target_feature, columns=["feature1", "feature3"])
        self.assertTrue(processed_feature.equals(target_feature))

        processor = BinaryDataProcessor(
            null_ratio_threshold=0,
            dominance_threshold=0.6,
            chi2_p_threshold=0.0001,
            impute_strategy="constant",
            impute_fill_value=0,
        )
        with self.assertRaises(RuntimeError):
            transformed_data = processor.fit_transform(data)


class TestContinuesDataProcessor(unittest.TestCase):
    """
    Test ContinuesDataProcessor class.
    """

    def test_init(self):
        """
        Test __init__ method.
        """
        processor = ContinuesDataProcessor()
        self.assertIsInstance(processor, ContinuesDataProcessor)
        self.assertIsInstance(processor, DataProcessorBase)
        self.assertEqual(processor.null_ratio_threshold, 0.5)
        self.assertEqual(processor.variance_threshold, 0.01)
        self.assertEqual(processor.impute_strategy, "mean")
        self.assertEqual(processor.fold_change_threshold, None)
        self.assertEqual(processor.target_corr_threshold, 0)
        self.assertEqual(processor.skew_threshold, 2.5)
        self.assertEqual(processor.skew_correction_method, "yeo-johnson")
        self.assertEqual(processor.outlier_detection_method, "z_score")
        self.assertEqual(processor.outlier_detection_threshold, 3)
        self.assertEqual(processor.outlier_correction_method, "clip")
        self.assertEqual(processor.scaling_method, "minmax")
        self.assertEqual(processor.feature_corr_threshold, 0.95)

        (
            high_quality_feature_filter,
            informative_feature_filter,
            missing_value_imputer,
            skewness_corrector,
            outlier_corrector,
            feature_scaler,
            corrected_feature_filter,
        ) = processor.processors

        self.assertIsInstance(high_quality_feature_filter, HighQualityFeatureFilter)
        self.assertEqual(high_quality_feature_filter.null_ratio_threshold, 0.5)
        self.assertEqual(high_quality_feature_filter.variance_threshold, 0.01)
        self.assertIsNone(high_quality_feature_filter.dominance_threshold)

        self.assertIsInstance(informative_feature_filter, InformativeFeatureFilter)
        self.assertEqual(informative_feature_filter.target_corr_threshold, 0)
        self.assertEqual(informative_feature_filter.fold_change_threshold, None)
        self.assertIsNone(informative_feature_filter.chi2_p_threshold)

        self.assertIsInstance(missing_value_imputer, MissingValueImputer)
        self.assertEqual(missing_value_imputer.impute_strategy, "mean")
        self.assertEqual(missing_value_imputer.impute_fill_value, 0)

        self.assertIsInstance(skewness_corrector, SkewnessCorrector)
        self.assertEqual(skewness_corrector.skew_threshold, 2.5)
        self.assertEqual(skewness_corrector.skew_correction_method, "yeo-johnson")

        self.assertIsInstance(outlier_corrector, OutlierCorrector)
        self.assertEqual(outlier_corrector.detection_method, "z_score")
        self.assertEqual(outlier_corrector.detection_threshold, 3)
        self.assertEqual(outlier_corrector.correction_method, "clip")

        self.assertIsInstance(feature_scaler, FeatureScaler)
        self.assertEqual(feature_scaler.scale_method, "minmax")
        self.assertEqual(feature_scaler.scale_range, (0, 1))

        self.assertIsInstance(corrected_feature_filter, CorrelatedFeatureFilter)
        self.assertEqual(corrected_feature_filter.feature_corr_threshold, 0.95)

        processor = ContinuesDataProcessor(
            null_ratio_threshold=0,
            variance_threshold=0,
            impute_strategy="median",
            fold_change_threshold=10,
            target_corr_threshold=0,
            skew_threshold=0.01,
            skew_correction_method="exp",
            outlier_detection_method="iqr",
            outlier_detection_threshold=1.001,
            outlier_correction_method="mean",
            scaling_method="minmax",
            scaling_feature_range=(-1, 1),
            feature_corr_threshold=0,
        )
        self.assertIsInstance(processor, ContinuesDataProcessor)
        self.assertIsInstance(processor, DataProcessorBase)
        self.assertEqual(processor.null_ratio_threshold, 0)
        self.assertEqual(processor.variance_threshold, 0)
        self.assertEqual(processor.impute_strategy, "median")
        self.assertEqual(processor.fold_change_threshold, 10)
        self.assertEqual(processor.target_corr_threshold, 0)
        self.assertEqual(processor.skew_threshold, 0.01)
        self.assertEqual(processor.skew_correction_method, "exp")
        self.assertEqual(processor.outlier_detection_method, "iqr")
        self.assertEqual(processor.outlier_detection_threshold, 1.001)
        self.assertEqual(processor.outlier_correction_method, "mean")
        self.assertEqual(processor.scaling_method, "minmax")
        self.assertEqual(processor.feature_corr_threshold, 0)

        (
            high_quality_feature_filter,
            informative_feature_filter,
            missing_value_imputer,
            skewness_corrector,
            outlier_corrector,
            feature_scaler,
            corrected_feature_filter,
        ) = processor.processors

        self.assertIsInstance(high_quality_feature_filter, HighQualityFeatureFilter)
        self.assertEqual(high_quality_feature_filter.null_ratio_threshold, 0)
        self.assertEqual(high_quality_feature_filter.variance_threshold, 0)

        self.assertIsInstance(informative_feature_filter, InformativeFeatureFilter)
        self.assertEqual(informative_feature_filter.target_corr_threshold, 0)
        self.assertEqual(informative_feature_filter.fold_change_threshold, 10)
        self.assertIsNone(informative_feature_filter.chi2_p_threshold)

        self.assertIsInstance(missing_value_imputer, MissingValueImputer)
        self.assertEqual(missing_value_imputer.impute_strategy, "median")
        self.assertEqual(missing_value_imputer.impute_fill_value, 0)

        self.assertIsInstance(skewness_corrector, SkewnessCorrector)
        self.assertEqual(skewness_corrector.skew_threshold, 0.01)
        self.assertEqual(skewness_corrector.skew_correction_method, "exp")

        self.assertIsInstance(outlier_corrector, OutlierCorrector)
        self.assertEqual(outlier_corrector.detection_method, "iqr")
        self.assertEqual(outlier_corrector.detection_threshold, 1.001)
        self.assertEqual(outlier_corrector.correction_method, "mean")

        self.assertIsInstance(feature_scaler, FeatureScaler)
        self.assertEqual(feature_scaler.scale_method, "minmax")
        self.assertEqual(feature_scaler.scale_range, (-1, 1))

        self.assertIsInstance(corrected_feature_filter, CorrelatedFeatureFilter)
        self.assertEqual(corrected_feature_filter.feature_corr_threshold, 0)

        processor = ContinuesDataProcessor(
            null_ratio_threshold=0.9999,
            variance_threshold=1000,
            impute_strategy="constant",
            impute_fill_value=0,
            fold_change_threshold=10,
            target_corr_threshold=1,
            skew_threshold=100,
            skew_correction_method="square",
            outlier_detection_method="modified_z_score",
            outlier_detection_threshold=100,
            outlier_correction_method="median",
            scaling_method="robust",
            scaling_feature_range=(-1, 1),
            feature_corr_threshold=1,
        )
        self.assertIsInstance(processor, ContinuesDataProcessor)
        self.assertIsInstance(processor, DataProcessorBase)
        self.assertEqual(processor.null_ratio_threshold, 0.9999)
        self.assertEqual(processor.variance_threshold, 1000)
        self.assertEqual(processor.impute_strategy, "constant")
        self.assertEqual(processor.impute_fill_value, 0)
        self.assertEqual(processor.fold_change_threshold, 10)
        self.assertEqual(processor.target_corr_threshold, 1)
        self.assertEqual(processor.skew_threshold, 100)
        self.assertEqual(processor.skew_correction_method, "square")
        self.assertEqual(processor.outlier_detection_method, "modified_z_score")
        self.assertEqual(processor.outlier_detection_threshold, 100)
        self.assertEqual(processor.outlier_correction_method, "median")
        self.assertEqual(processor.scaling_method, "robust")
        self.assertEqual(processor.feature_corr_threshold, 1)

        (
            high_quality_feature_filter,
            informative_feature_filter,
            missing_value_imputer,
            skewness_corrector,
            outlier_corrector,
            feature_scaler,
            corrected_feature_filter,
        ) = processor.processors

        self.assertIsInstance(high_quality_feature_filter, HighQualityFeatureFilter)
        self.assertEqual(high_quality_feature_filter.null_ratio_threshold, 0.9999)
        self.assertEqual(high_quality_feature_filter.variance_threshold, 1000)

        self.assertIsInstance(informative_feature_filter, InformativeFeatureFilter)
        self.assertEqual(informative_feature_filter.target_corr_threshold, 1)
        self.assertEqual(informative_feature_filter.fold_change_threshold, 10)
        self.assertIsNone(informative_feature_filter.chi2_p_threshold)

        self.assertIsInstance(missing_value_imputer, MissingValueImputer)
        self.assertEqual(missing_value_imputer.impute_strategy, "constant")
        self.assertEqual(missing_value_imputer.impute_fill_value, 0)

        self.assertIsInstance(skewness_corrector, SkewnessCorrector)
        self.assertEqual(skewness_corrector.skew_threshold, 100)
        self.assertEqual(skewness_corrector.skew_correction_method, "square")

        self.assertIsInstance(outlier_corrector, OutlierCorrector)
        self.assertEqual(outlier_corrector.detection_method, "modified_z_score")
        self.assertEqual(outlier_corrector.detection_threshold, 100)
        self.assertEqual(outlier_corrector.correction_method, "median")

        self.assertIsInstance(feature_scaler, FeatureScaler)
        self.assertEqual(feature_scaler.scale_method, "robust")
        self.assertEqual(feature_scaler.scale_range, None)

        self.assertIsInstance(corrected_feature_filter, CorrelatedFeatureFilter)
        self.assertEqual(corrected_feature_filter.feature_corr_threshold, 1)

        with self.assertRaises(TypeError):
            processor = ContinuesDataProcessor(null_ratio_threshold="1")
        with self.assertRaises(ValueError):
            processor = ContinuesDataProcessor(null_ratio_threshold=-0.1)
        with self.assertRaises(ValueError):
            processor = ContinuesDataProcessor(null_ratio_threshold=1)
        with self.assertRaises(TypeError):
            processor = ContinuesDataProcessor(variance_threshold="1")
        with self.assertRaises(ValueError):
            processor = ContinuesDataProcessor(variance_threshold=-0.1)
        with self.assertRaises(ValueError):
            processor = ContinuesDataProcessor(impute_strategy="most_frequent")
        with self.assertRaises(ValueError):
            processor = ContinuesDataProcessor(fold_change_threshold=-0.1)
        with self.assertRaises(ValueError):
            processor = ContinuesDataProcessor(target_corr_threshold=-0.1)
        with self.assertRaises(ValueError):
            processor = ContinuesDataProcessor(target_corr_threshold=1.1)
        with self.assertRaises(ValueError):
            processor = ContinuesDataProcessor(skew_threshold=-1)
        with self.assertRaises(ValueError):
            processor = ContinuesDataProcessor(skew_correction_method="test")
        with self.assertRaises(ValueError):
            processor = ContinuesDataProcessor(outlier_detection_method="test")
        with self.assertRaises(ValueError):
            processor = ContinuesDataProcessor(outlier_detection_threshold=-1)
        with self.assertRaises(ValueError):
            processor = ContinuesDataProcessor(outlier_correction_method="test")
        with self.assertRaises(ValueError):
            processor = ContinuesDataProcessor(scaling_method="test")
        with self.assertRaises(TypeError):
            processor = ContinuesDataProcessor(scaling_feature_range=1)
        with self.assertRaises(ValueError):
            processor = ContinuesDataProcessor(scaling_feature_range=[1, -1])
        with self.assertRaises(ValueError):
            processor = ContinuesDataProcessor(feature_corr_threshold=-0.1)
        with self.assertRaises(ValueError):
            processor = ContinuesDataProcessor(feature_corr_threshold=1.1)

    def test_fit(self):
        """
        Test fit method.
        """
        X = pd.DataFrame(
            {
                "feature1": [0, 1, 0, 1, 0, np.nan],
                "feature2": [1, 0, 1, 0, 1, np.nan],
                "feature3": [0, 0, 1, 1, 0, np.nan],
                "feature4": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan],
                "feature5": [0.1, 10, 0.2, 5, 0.3, np.nan],
                "feature6": [-10, 0.1, -5, 0.2, -20, np.nan],
                "feature7": [1, 2, 3, 4, 5, -1000],
                "feature8": [4, 5, 5, 5, 5, 10000],
                "feature9": [1, 2, 3, 4, 5, 6],
                "feature10": [100, 100, 100, 100, 100, 1],
                "feature11": [1, 2, 3, 4, 5, 10000],
                "feature12": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0, 1])
        data = DataBase(features=X, labels=y)

        processor = ContinuesDataProcessor(scaling_method="standard")
        processor.fit(data)
        (
            high_quality_feature_filter,
            informative_feature_filter,
            missing_value_imputer,
            skewness_corrector,
            outlier_corrector,
            feature_scaler,
            corrected_feature_filter,
        ) = processor.processors
        self.assertListEqual(
            high_quality_feature_filter.selected_features_.tolist(),
            [
                "feature1",
                "feature2",
                "feature3",
                "feature4",
                "feature5",
                "feature6",
                "feature7",
                "feature8",
                "feature9",
                "feature10",
                "feature11",
            ],
        )
        self.assertListEqual(
            informative_feature_filter.identified_features_.tolist(),
            [
                "feature1",
                "feature2",
                "feature3",
                "feature4",
                "feature5",
                "feature6",
                "feature7",
                "feature8",
                "feature9",
                "feature10",
                "feature11",
            ],
        )
        self.assertEqual(
            missing_value_imputer.impute_values_,
            {
                "feature1": X["feature1"].mean(),
                "feature2": X["feature2"].mean(),
                "feature3": X["feature3"].mean(),
                "feature4": X["feature4"].mean(),
                "feature5": X["feature5"].mean(),
                "feature6": X["feature6"].mean(),
                "feature7": X["feature7"].mean(),
                "feature8": X["feature8"].mean(),
                "feature9": X["feature9"].mean(),
                "feature10": X["feature10"].mean(),
                "feature11": X["feature11"].mean(),
            },
        )
        self.assertEqual(skewness_corrector.left_skewed_features_, [])
        self.assertEqual(skewness_corrector.right_skewed_features_, [])
        target_outliers = {
            "feature1": pd.Series([False, False, False, False, False, False]),
            "feature2": pd.Series([False, False, False, False, False, False]),
            "feature3": pd.Series([False, False, False, False, False, False]),
            "feature4": pd.Series([False, False, False, False, False, False]),
            "feature5": pd.Series([False, False, False, False, False, False]),
            "feature6": pd.Series([False, False, False, False, False, False]),
            "feature7": pd.Series([False, False, False, False, False, False]),
            "feature8": pd.Series([False, False, False, False, False, False]),
            "feature9": pd.Series([False, False, False, False, False, False]),
            "feature10": pd.Series([False, False, False, False, False, False]),
            "feature11": pd.Series([False, False, False, False, False, False]),
        }
        self.assertEqual(outlier_corrector.outliers_.keys(), target_outliers.keys())
        for key, value in outlier_corrector.outliers_.items():
            self.assertTrue(value.equals(target_outliers[key]))
        self.assertIsInstance(feature_scaler.scaler_, StandardScaler)
        self.assertSetEqual(
            set(corrected_feature_filter.dropped_features_),
            {"feature1", "feature7", "feature10", "feature11"},
        )
        processor = ContinuesDataProcessor(
            null_ratio_threshold=0.5,
            variance_threshold=0.1,
            impute_strategy="mean",
            fold_change_threshold=1.5,
            target_corr_threshold=0,
            skew_threshold=2,
            skew_correction_method="yeo-johnson",
            outlier_detection_method="z_score",
            outlier_detection_threshold=2,
            outlier_correction_method="clip",
            scaling_method="minmax",
            feature_corr_threshold=0.95,
        )
        processor.fit(data)

        (
            high_quality_feature_filter,
            informative_feature_filter,
            missing_value_imputer,
            skewness_corrector,
            outlier_corrector,
            feature_scaler,
            corrected_feature_filter,
        ) = processor.processors
        self.assertListEqual(
            high_quality_feature_filter.selected_features_.tolist(),
            [
                "feature1",
                "feature2",
                "feature3",
                "feature4",
                "feature5",
                "feature6",
                "feature7",
                "feature8",
                "feature9",
                "feature10",
                "feature11",
            ],
        )
        self.assertListEqual(
            informative_feature_filter.identified_features_.tolist(),
            [
                "feature1",
                "feature2",
                "feature3",
                "feature5",
                "feature6",
                "feature7",
                "feature8",
                "feature10",
                "feature11",
            ],
        )
        self.assertEqual(
            missing_value_imputer.impute_values_,
            {
                "feature1": X["feature1"].mean(),
                "feature2": X["feature2"].mean(),
                "feature3": X["feature3"].mean(),
                "feature5": X["feature5"].mean(),
                "feature6": X["feature6"].mean(),
                "feature7": X["feature7"].mean(),
                "feature8": X["feature8"].mean(),
                "feature10": X["feature10"].mean(),
                "feature11": X["feature11"].mean(),
            },
        )
        self.assertEqual(skewness_corrector.left_skewed_features_, ["feature10"])
        self.assertEqual(
            skewness_corrector.right_skewed_features_, ["feature8", "feature11"]
        )
        target_outliers = {
            "feature1": pd.Series([False, False, False, False, False, False]),
            "feature2": pd.Series([False, False, False, False, False, False]),
            "feature3": pd.Series([False, False, False, False, False, False]),
            "feature5": pd.Series([False, False, False, False, False, False]),
            "feature6": pd.Series([False, False, False, False, False, False]),
            "feature7": pd.Series([False, False, False, False, False, True]),
            "feature8": pd.Series([True, False, False, False, False, False]),
            "feature10": pd.Series([False, False, False, False, False, True]),
            "feature11": pd.Series([False, False, False, False, False, False]),
        }
        self.assertEqual(outlier_corrector.outliers_.keys(), target_outliers.keys())
        for key, value in outlier_corrector.outliers_.items():
            self.assertTrue(value.equals(target_outliers[key]))
        self.assertIsInstance(feature_scaler.scaler_, MinMaxScaler)
        self.assertSetEqual(
            set(corrected_feature_filter.dropped_features_), {"feature2", "feature7"},
        )

    def test_transform(self):
        """
        Test transform method.
        """
        X = pd.DataFrame(
            {
                "feature1": [0, 1, 0, 1, 0, np.nan],
                "feature2": [1, 0, 1, 0, 1, np.nan],
                "feature3": [0, 0, 1, 1, 0, np.nan],
                "feature4": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan],
                "feature5": [0.1, 10, 0.2, 5, 0.3, np.nan],
                "feature6": [-10, 0.1, -5, 0.2, -20, np.nan],
                "feature7": [1, 2, 3, 4, 5, -1000],
                "feature8": [4, 5, 5, 5, 5, 10000],
                "feature9": [1, 2, 3, 4, 5, 6],
                "feature10": [100, 100, 100, 100, 100, 1],
                "feature11": [1, 2, 3, 4, 5, 10000],
                "feature12": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0, 1])
        data = DataBase(features=X, labels=y)

        processor = ContinuesDataProcessor()
        with self.assertRaises(ValueError):
            processor.transform(data)
        processor.fit(data)
        transformed_data = processor.transform(data)
        self.assertIsInstance(transformed_data, DataBase)
        raw_feature = transformed_data.get_features(layer="raw")
        processed_feature = transformed_data.get_features(layer="processed").astype(
            float
        )
        self.assertTrue(raw_feature.equals(X))
        self.assertListEqual(
            processed_feature.columns.tolist(),
            [
                "feature1",
                "feature3",
                "feature4",
                "feature5",
                "feature6",
                "feature8",
                "feature9",
            ],
        )

        with self.assertRaises(ValueError):
            processor.transform(DataBase(features=X.loc[:, ["feature1", "feature2"]]))
        with self.assertRaises(TypeError):
            processor.transform(X)

    def test_fit_transform(self):
        """
        Test fit_transform method.
        """
        X = pd.DataFrame(
            {
                "feature1": [0, 1, 0, 1, 0, np.nan],
                "feature2": [1, 0, 1, 0, 1, np.nan],
                "feature3": [0, 0, 1, 1, 0, np.nan],
                "feature4": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan],
                "feature5": [0.1, 10, 0.2, 5, 0.3, np.nan],
                "feature6": [-10, 0.1, -5, 0.2, -20, np.nan],
                "feature7": [1, 2, 3, 4, 5, -1000],
                "feature8": [4, 5, 5, 5, 5, 10000],
                "feature9": [1, 2, 3, 4, 5, 6],
                "feature10": [100, 100, 100, 100, 100, 1],
                "feature11": [1, 2, 3, 4, 5, 10000],
                "feature12": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0, 1])
        data = DataBase(features=X, labels=y)

        processor = ContinuesDataProcessor()
        transformed_data = processor.fit_transform(data)
        raw_feature = transformed_data.get_features(layer="raw")
        processed_feature = transformed_data.get_features(layer="processed").astype(
            float
        )
        self.assertTrue(raw_feature.equals(X))
        self.assertListEqual(
            processed_feature.columns.tolist(),
            [
                "feature1",
                "feature3",
                "feature4",
                "feature5",
                "feature6",
                "feature8",
                "feature9",
            ],
        )
        with self.assertRaises(TypeError):
            processor.transform(X)


def main() -> None:
    unittest.main()


if __name__ == "__main__":
    main()
