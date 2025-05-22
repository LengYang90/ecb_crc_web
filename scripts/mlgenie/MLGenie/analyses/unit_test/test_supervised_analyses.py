#!/usr/bin/env python3
###############################################################
## Copyright: PromptBio Corp 2024
# Author: jiayuchen
# Date of creation: 09/29/2024
# Date of revision: 01/06/2025
#
## MLGenie
## Description: Unit test for SupervisedAnalysis
#
###############################################################

import unittest
import os
import sys
from typing import List, Dict


import numpy as np
import pandas as pd
import random
import scipy
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

cur_folder = os.path.dirname(os.path.realpath(__file__))
last_folder = os.path.dirname(cur_folder)
last_last_folder = os.path.dirname(last_folder)
last_last_last_folder = os.path.dirname(last_last_folder)
last_last_last_last_folder = os.path.dirname(last_last_last_folder)
sys.path.append(last_last_folder)
sys.path.append(last_last_last_folder)
sys.path.append(last_last_last_last_folder)

from MLGenie.analyses.supervised_analyses import SupervisedAnalysis, ClsAnalysis
from MLGenie.Data import MultiOmicsData, DataBase
from MLGenie.utils import Metrics, HPOAlgorithm, prepare_train_test_data
from MLGenie.model.classification import ClsModelType


class TestSupervisedAnalysis(unittest.TestCase):
    """ 
    Test the SupervisedAnalysis class
    """
    def setUp(self):
        # Create sample features and labels for binary classification
        np.random.seed(42)
        self.features = pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_{i}' for i in range(10)])
        self.labels = pd.Series(np.random.randint(0, 2, 100), name='label')
        self.data = DataBase(features=[self.features], labels=self.labels)

    def test_check_parameters(self):
        # Test with valid parameters
        analysis = SupervisedAnalysis()
        analysis._check_parameters(
            metrics=Metrics.AUC,
            n_feature_to_select=10,
            specified_models=["model1", "model2"],
            hpo_algorithm=HPOAlgorithm.GridSearch,
            hpo_search_iter=100,
            cv=5,
            n_jobs=8,
        )
        self.assertTrue(True)  # No exception should be raised

        # Test with invalid metrics
        with self.assertRaises(TypeError):
            analysis._check_parameters(
                metrics="invalid_metrics",
                n_feature_to_select=10,
                specified_models=["model1", "model2"],
                hpo_algorithm=HPOAlgorithm.GridSearch,
                hpo_search_iter=100,
                cv=5,
                n_jobs=8,
            )

        # Test with invalid n_feature_to_select
        with self.assertRaises(ValueError):
            analysis._check_parameters(
                metrics=Metrics.AUC,
                n_feature_to_select=-1,
                specified_models=["model1", "model2"],
                hpo_algorithm=HPOAlgorithm.GridSearch,
                hpo_search_iter=100,
                cv=5,
                n_jobs=8,
            )

        # Test with invalid specified_models
        with self.assertRaises(ValueError):
            analysis._check_parameters(
                metrics=Metrics.AUC,
                n_feature_to_select=10,
                specified_models=["model1", 2],
                hpo_algorithm=HPOAlgorithm.GridSearch,
                hpo_search_iter=100,
                cv=5,
                n_jobs=8,
            )

        # Test with invalid hpo_algorithm
        with self.assertRaises(TypeError):
            analysis._check_parameters(
                metrics=Metrics.AUC,
                n_feature_to_select=10,
                specified_models=["model1", "model2"],
                hpo_algorithm="invalid_algorithm",
                hpo_search_iter=100,
                cv=5,
                n_jobs=8,
            )

        # Test with invalid hpo_search_iter
        with self.assertRaises(ValueError):
            analysis._check_parameters(
                metrics=Metrics.AUC,
                n_feature_to_select=10,
                specified_models=["model1", "model2"],
                hpo_algorithm=HPOAlgorithm.GridSearch,
                hpo_search_iter=0,
                cv=5,
                n_jobs=8,
            )

        # Test with invalid cv
        with self.assertRaises(ValueError):
            analysis._check_parameters(
                metrics=Metrics.AUC,
                n_feature_to_select=10,
                specified_models=["model1", "model2"],
                hpo_algorithm=HPOAlgorithm.GridSearch,
                hpo_search_iter=100,
                cv=1,
                n_jobs=8,
            )

        # Test with invalid n_jobs
        with self.assertRaises(ValueError):
            analysis._check_parameters(
                metrics=Metrics.AUC,
                n_feature_to_select=10,
                specified_models=["model1", "model2"],
                hpo_algorithm=HPOAlgorithm.GridSearch,
                hpo_search_iter=100,
                cv=5,
                n_jobs=0,
            )

    def test_batch_effect_normalization(self):
        # Test batch effect normalization with more complex data
        features = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            'batch': ['A', 'A', 'B', 'B', 'C', 'C']
        })
        labels = pd.Series([0, 1, 0, 1, 0, 1])
        data = DataBase(features=[features], labels=labels)
        analysis = SupervisedAnalysis(result_dir="test_dir", random_state=42)
        normalized_data = analysis._batch_effect_normalization(data)

        # Check if the batch column is removed from the normalized data
        self.assertNotIn('batch', normalized_data.get_features("processed").columns)

        # Check if the other features are still present
        self.assertIn('feature1', normalized_data.get_features("processed").columns)
        self.assertIn('feature2', normalized_data.get_features("processed").columns)

        # Check if the number of samples remains the same
        self.assertEqual(len(normalized_data.get_features("processed")), 6)

        # Check if the labels are unchanged
        pd.testing.assert_series_equal(normalized_data.get_labels(), labels)

        # Check if the raw features are unchanged
        pd.testing.assert_frame_equal(normalized_data.get_features("raw"), features)

        # Check if the processed features are different from the raw features
        self.assertFalse(normalized_data.get_features("processed").equals(features.drop(columns=['batch'])))

        # Test batch effect normalization when there's no batch column
        features = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        labels = pd.Series([0, 1, 0, 1, 0])
        data = DataBase(features=[features], labels=labels)

        analysis = SupervisedAnalysis(result_dir="test_dir", random_state=42)
        normalized_data = analysis._batch_effect_normalization(data)

        # Check if the features remain unchanged when there's no batch column
        pd.testing.assert_frame_equal(normalized_data.get_features("raw"), features)
        pd.testing.assert_frame_equal(normalized_data.get_features("processed"), features)



class TestClsAnalysis(unittest.TestCase):
    def setUp(self):
        # Create sample features and labels for binary classification
        np.random.seed(42)
        self.features = pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_{i}' for i in range(10)])
        self.labels = pd.Series(np.random.randint(0, 2, 100), name='label')
        self.data = DataBase(features=[self.features], labels=self.labels)

    def test_init(self):
        # Test initialization with valid control_group
        cls_analysis = ClsAnalysis(control_group=0, result_dir="test_dir", random_state=42)
        self.assertEqual(cls_analysis.control_group, 0)

        # Test initialization with string control_group
        cls_analysis = ClsAnalysis(control_group="control", result_dir="test_dir", random_state=42)
        self.assertEqual(cls_analysis.control_group, "control")

        # Test initialization without control_group
        cls_analysis = ClsAnalysis(result_dir="test_dir", random_state=42)
        self.assertIsNone(cls_analysis.control_group)

        # Test initialization with invalid control_group type
        with self.assertRaises(TypeError):
            ClsAnalysis(control_group=1.5, result_dir="test_dir", random_state=42)

    def test_check_control_group(self):
        # Test with valid control_group
        cls_analysis = ClsAnalysis(control_group=0, result_dir="test_dir", random_state=42)
        cls_analysis._check_control_group(self.data)  # Should not raise an error

        # Test with invalid control_group
        cls_analysis = ClsAnalysis(control_group=2, result_dir="test_dir", random_state=42)
        with self.assertRaises(ValueError):
            cls_analysis._check_control_group(self.data)

        # Test with string control_group
        labels = pd.Series(['A', 'B', 'A', 'B'], name='label')
        data = DataBase(features=[pd.DataFrame({'f1': [1, 2, 3, 4]})], labels=labels)
        cls_analysis = ClsAnalysis(control_group='A', result_dir="test_dir", random_state=42)
        cls_analysis._check_control_group(data)  # Should not raise an error

        cls_analysis = ClsAnalysis(control_group='C', result_dir="test_dir", random_state=42)
        with self.assertRaises(ValueError):
            cls_analysis._check_control_group(data)

        # Test when control_group is None and labels are binary (0 and 1)
        binary_labels = pd.Series([0, 1, 0, 1], name='label')
        binary_data = DataBase(features=[pd.DataFrame({'f1': [1, 2, 3, 4]})], labels=binary_labels)
        cls_analysis = ClsAnalysis(control_group=None, result_dir="test_dir", random_state=42)
        cls_analysis._check_control_group(binary_data)  # Should not raise an error

        # Test when control_group is None and labels are not binary
        non_binary_labels = pd.Series([0, 1, 2, 3], name='label')
        non_binary_data = DataBase(features=[pd.DataFrame({'f1': [1, 2, 3, 4]})], labels=non_binary_labels)
        with self.assertRaises(ValueError):
            cls_analysis._check_control_group(non_binary_data)

        # Test when control_group is None and labels are binary but not 0 and 1
        other_binary_labels = pd.Series([2, 3, 2, 3], name='label')
        other_binary_data = DataBase(features=[pd.DataFrame({'f1': [1, 2, 3, 4]})], labels=other_binary_labels)
        with self.assertRaises(ValueError):
            cls_analysis._check_control_group(other_binary_data)

    def test_convert_label_to_binary(self):
        # Test with control_group as 0
        cls_analysis = ClsAnalysis(control_group=0, result_dir="test_dir", random_state=42)
        labels = pd.Series([0, 1, 2, 0, 1], name='label')
        data = DataBase(features=[pd.DataFrame({'f1': [1, 2, 3, 4, 5]})], labels=labels)
        converted_data = cls_analysis._convert_label_to_binary(data)
        expected_labels = pd.Series([0, 1, 1, 0, 1], name='label')
        pd.testing.assert_series_equal(converted_data.get_labels(), expected_labels)

        # Test with control_group as string
        cls_analysis = ClsAnalysis(control_group='A', result_dir="test_dir", random_state=42)
        labels = pd.Series(['A', 'B', 'C', 'A', 'B'], name='label')
        data = DataBase(features=[pd.DataFrame({'f1': [1, 2, 3, 4, 5]})], labels=labels)
        converted_data = cls_analysis._convert_label_to_binary(data)
        expected_labels = pd.Series([0, 1, 1, 0, 1], name='label')
        pd.testing.assert_series_equal(converted_data.get_labels(), expected_labels)

        # Test with control_group as None (should not change labels)
        cls_analysis = ClsAnalysis(control_group=None, result_dir="test_dir", random_state=42)
        labels = pd.Series([0, 1, 0, 1], name='label')
        data = DataBase(features=[pd.DataFrame({'f1': [1, 2, 3, 4]})], labels=labels)
        converted_data = cls_analysis._convert_label_to_binary(data)
        pd.testing.assert_series_equal(converted_data.get_labels(), labels)


    def test_feature_selection(self):
        # Create a sample dataset
        data = DataBase(features=[self.features], labels=self.labels, if_processed=True)

        # Initialize ClsAnalysis with specific parameters
        cls_analysis = ClsAnalysis(
            n_feature_to_select=10,
            result_dir="test_dir",
            random_state=42
        )

        # Perform feature selection
        selected_features, selected_scores = cls_analysis._feature_selection(data)

        # Check if the correct number of features were selected
        self.assertEqual(len(selected_features), 10)
        self.assertEqual(len(selected_scores), 10)

        # Check if the selected features are from the original feature set
        self.assertTrue(all(feature in self.features.columns for feature in selected_features))

        # Create a sample dataset with fewer features than n_feature_to_select
        small_features = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature_{i}' for i in range(5)])
        labels = pd.Series(np.random.randint(0, 2, 100), name='label')
        data = DataBase(features=[small_features], labels=labels, if_processed=True)

        # Initialize ClsAnalysis with n_feature_to_select greater than available features
        cls_analysis = ClsAnalysis(
            n_feature_to_select=10,
            result_dir="test_dir",
            random_state=42
        )
        # Check if ValueError is raised when performing feature selection
        with self.assertRaises(ValueError) as context:
            cls_analysis._feature_selection(data)

        # Check if the error message is as expected
        self.assertTrue("The number of features to select cannot exceed the number of features in the data." in str(context.exception))

    def test_model_selection(self):
        # Create a sample dataset
        features = pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_{i}' for i in range(10)])
        labels = pd.Series(np.random.randint(0, 2, 100), name='label')
        data = DataBase(features=[features], labels=labels, if_processed=True)

        # Initialize ClsAnalysis with specific parameters
        specified_models = ["SVM", "LR"]
        cls_analysis = ClsAnalysis(
            result_dir="test_dir",
            random_state=42,
            specified_models=specified_models,
            metrics=Metrics.AUC,
            cv=3,
            n_jobs=1
        )

        # Perform model selection
        model_comparison, selected_model, best_metric = cls_analysis._model_selection(data)
        # Check if model_comparison is a list of tuples
        self.assertIsInstance(model_comparison, list)
        self.assertTrue(all(isinstance(item, tuple) for item in model_comparison))

        # Check if each tuple in model_comparison has two elements
        self.assertTrue(all(len(item) == 2 for item in model_comparison))

        # Check if the second element of each tuple is a float (metric score)
        self.assertTrue(all(isinstance(item[1], float) for item in model_comparison))

        # Check if model_comparison contains entries for all specified models
        self.assertEqual(len(model_comparison), len(specified_models))
        self.assertTrue(all(any(model == spec_model for model, _ in model_comparison) for spec_model in specified_models))

        # Check if the models are sorted by their metric scores in descending order
        self.assertEqual(model_comparison, sorted(model_comparison, key=lambda x: x[1], reverse=True))

        # Check if the best_metric matches the highest score in model_comparison
        self.assertEqual(best_metric, max(score for _, score in model_comparison))

        # Check if the selected model is an instance of ClsModelType
        self.assertIsInstance(selected_model, ClsModelType)

        # Check if the selected model is one of the specified models
        self.assertIn(selected_model.name, specified_models)

        # Test with different metrics
        cls_analysis.metrics = Metrics.Accuracy
        _, selected_model_accuracy, best_metric_accuracy = cls_analysis._model_selection(data)
        self.assertIsInstance(selected_model_accuracy, ClsModelType)

        # Test with no specified models (should use all available models)
        cls_analysis.specified_models = None
        _, selected_model_all, best_metric_all = cls_analysis._model_selection(data)
        self.assertIsInstance(selected_model_all, ClsModelType)

    def test_compute_feature_shap_values(self):
        features = pd.DataFrame(np.random.rand(100, 20), columns=[f'feature_{i}' for i in range(20)])
        labels = pd.Series(np.random.randint(0, 2, 100), name='label')
        data = DataBase(features=[features], labels=labels)

        features = pd.DataFrame(np.random.rand(100, 20), columns=[f'feature_{i}' for i in range(20)])
        labels = pd.Series(np.random.randint(0, 2, 100), name='label')
        test_data = DataBase(features=[features], labels=labels)

        # Initialize ClsAnalysis with specific parameters
        specified_models = ["SVM", "LR"]
        cls_analysis = ClsAnalysis(
            result_dir="test_save_shap_dir",
            random_state=42,
            metrics=Metrics.AUC,
            n_feature_to_select = 4,
            cv=5,
            n_jobs=1,
            specified_models=specified_models
        )

        # Fit the model to simulate the SHAP values computation
        cls_analysis.fit(data)
        cls_analysis.transform(test_data)

        # Test if SHAP values are computed and saved correctly
        shap_dir = os.path.join(cls_analysis.result_dir, 'shap')
        self.assertTrue(os.path.exists(shap_dir))

        # Check if the SHAP summary plot is saved
        summary_plot_path = os.path.join(shap_dir, 'shap_summary_plot.png')
        self.assertTrue(os.path.exists(summary_plot_path))

        # Check if the SHAP stripplot offsets are saved
        offsets_csv_path = os.path.join(shap_dir, 'shap_stripplot_offsets.csv')
        self.assertTrue(os.path.exists(offsets_csv_path))

        # Check if the SHAP dependence plots are saved for each feature
        for feature in cls_analysis.selected_features:
            dependence_plot_path = os.path.join(shap_dir, f'shap_scatter_plot_{feature}.png')
            self.assertTrue(os.path.exists(dependence_plot_path))

        # Check if the SHAP data is saved
        shap_data_csv_path = os.path.join(shap_dir, 'shap_data.csv')
        self.assertTrue(os.path.exists(shap_data_csv_path))


    def test_fit(self):
        # Create a sample dataset
        features = pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_{i}' for i in range(10)])
        labels = pd.Series(np.random.randint(0, 2, 100), name='label')
        data = DataBase(features=[features], labels=labels)

        # Initialize ClsAnalysis with specific parameters
        specified_models = ["SVM", "LR"]
        cls_analysis = ClsAnalysis(
            specified_models = specified_models,
            result_dir="test_dir",
            random_state=123,
            n_feature_to_select=5,
            metrics=Metrics.AUC,
            cv=5,
            n_jobs=1
        )

        # Fit the model
        cls_analysis.fit(data)

        # Check if selected_features are created
        self.assertIsInstance(cls_analysis.selected_features, list)
        self.assertEqual(len(cls_analysis.selected_features), 5)

        # Check if model is created and fitted
        self.assertIsNotNone(cls_analysis.model)
        self.assertTrue(hasattr(cls_analysis.model, 'predict'))

        # Create a sample dataset with a control group
        features = pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_{i}' for i in range(10)])
        labels = pd.Series(['A' if i < 50 else 'B' for i in range(100)], name='label')
        data = DataBase(features=[features], labels=labels)

        # Initialize ClsAnalysis with specific parameters and a control group
        cls_analysis = ClsAnalysis(
            result_dir="test_dir",
            random_state=123,
            n_feature_to_select=10,
            metrics=Metrics.AUC,
            cv=5,
            n_jobs=1,
            control_group='A'
        )

        # Fit the model
        cls_analysis.fit(data)

        # Check if selected_features are created
        self.assertIsInstance(cls_analysis.selected_features, list)
        self.assertEqual(len(cls_analysis.selected_features), 10)

        # Check if model is created and fitted
        self.assertIsNotNone(cls_analysis.model)
        self.assertTrue(hasattr(cls_analysis.model, 'predict'))

        # Check if the control group is correctly handled
        self.assertEqual(cls_analysis.control_group, 'A')

    def test_transform(self):
        # Create a sample dataset
        features = pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_{i}' for i in range(10)])
        labels = pd.Series(np.random.randint(0, 2, 100), name='label')
        data = DataBase(features=[features], labels=labels, if_processed=True)

        train_data, test_data = prepare_train_test_data(data, test_ratio=0.2, random_state=123)

        # Initialize ClsAnalysis with specific parameters
        specified_models = ["SVM", "LR"]
        cls_analysis = ClsAnalysis(
            specified_models = specified_models,
            result_dir="test_dir",
            random_state=123,
            n_feature_to_select=5,
            metrics=Metrics.AUC,
            cv=5,
            n_jobs=2
        )

        # Fit the model
        cls_analysis.fit(train_data)

        # Test transform method
        (
            train_performance,
            train_ROC_data,
            train_PR_data
        ) = cls_analysis.transform(train_data)

        (
            test_performance,
            test_ROC_data,
            test_PR_data
        ) = cls_analysis.transform(test_data)

        # print("train_performance:",train_performance)
        # print("test_performance:",test_performance)
        # print("train_ROC_data:",train_ROC_data)
        # print("test_ROC_data:",test_ROC_data)
        # print("train_PR_data:",train_PR_data)
        # print("test_PR_data:",test_PR_data)

        # Check if the number of selected features matches n_feature_to_select
        self.assertEqual(len(cls_analysis.selected_features), cls_analysis.n_feature_to_select)

        # Check if all selected features are present in the original feature set
        original_features = data.get_feature_names()
        self.assertTrue(all(feature in original_features for feature in cls_analysis.selected_features))

        # Check if selected_scores is a list of floats with the same length as selected_features
        self.assertTrue(all(isinstance(score, float) for score in cls_analysis.selected_scores))
        self.assertEqual(len(cls_analysis.selected_scores), len(cls_analysis.selected_features))


        # Check if selected_scores are in descending order (higher scores first)
        selected_scores = cls_analysis.selected_scores.tolist()
        self.assertEqual(selected_scores, sorted(selected_scores, reverse=True))

        # Check if model_comparison is a list of tuples
        self.assertIsInstance(cls_analysis.model_comparison, list)
        self.assertTrue(all(isinstance(item, tuple) for item in cls_analysis.model_comparison))

        # Check if each tuple in model_comparison has two elements
        self.assertTrue(all(len(item) == 2 for item in cls_analysis.model_comparison))

        # Check if the second element of each tuple is a float (metric value)
        self.assertTrue(all(isinstance(item[1], float) for item in cls_analysis.model_comparison))

        # Check if the metric values are in descending order
        metric_values = [item[1] for item in cls_analysis.model_comparison]
        self.assertEqual(metric_values, sorted(metric_values, reverse=True))

        # Check if performance is created and contains expected keys
        self.assertIsInstance(train_performance, dict)
        expected_keys = [
            "auc", "accuracy", "sensitivity", "specificity",
            "confusion_matrix", "pr_auc", "precision", "recall",
            "f1_score",
        ]
        for key in expected_keys:
            self.assertIn(key, train_performance)
            self.assertIn(key, test_performance)

        # Check if performance values are of correct type and within expected range
        for key, value in train_performance.items():
            if "confusion_matrix" in key:
                self.assertIsInstance(value, tuple)
                self.assertEqual(len(value), 4)  # Assuming binary classification
            else:
                self.assertIsInstance(value, float)
                self.assertGreaterEqual(value, 0)
                self.assertLessEqual(value, 1)

        # Additional checks for specific metrics
        self.assertEqual(
            sum(train_performance['confusion_matrix']),
            len(train_data.get_labels())
        )
        self.assertEqual(
            sum(test_performance['confusion_matrix']),
            len(test_data.get_labels())
        )


    def test_fit_transform_with_MultiOmicsData(self):
        # Create sample MultiOmicsData
        np.random.seed(42)
        features_omics1 = pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_omics1_{i}' for i in range(10)])
        features_omics2 = pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_omics2_{i}' for i in range(10)])
        binary_dataframe = pd.DataFrame(np.random.randint(0, 2, (100, 10)), columns=[f'binary_feature_{i}' for i in range(10)])
        labels = pd.Series(np.random.randint(0, 2, 100), name='label')
        multi_omics_data = MultiOmicsData(
            gene_mutation=binary_dataframe,
            gene_expression=features_omics1,
            mirna_expression=features_omics2,
            labels=labels,
        )

        # Initialize the analysis
        specified_models = ["SVM", "LR"]
        analysis = ClsAnalysis(
            specified_models = specified_models,
            result_dir="test_dir",
            random_state=123,
            n_feature_to_select=10,
            metrics=Metrics.AUC,
            cv=5,
            n_jobs=1,
        )

        # Fit the analysis
        analysis.fit(multi_omics_data)

        # Transform the data
        performance, ROC_data, PR_data = analysis.transform(multi_omics_data)

        # Check if performance is a dictionary
        self.assertIsInstance(performance, dict)

        # Check if ROC_data and PR_data are tuples
        self.assertIsInstance(ROC_data, tuple)
        self.assertIsInstance(PR_data, tuple)

        # Check if the expected keys are present in the performance dictionary
        expected_keys = [
            "auc", "accuracy", "sensitivity", "specificity",
            "confusion_matrix", "pr_auc", "precision", "recall",
            "f1_score",
        ]
        for key in expected_keys:
            self.assertIn(key, performance)

        # Check if the values in the performance dictionary are of the correct type
        for key, value in performance.items():
            if key == "confusion_matrix":
                self.assertIsInstance(value, tuple)
                self.assertEqual(len(value), 4)  # Assuming binary classification
            else:
                self.assertIsInstance(value, float)
                self.assertGreaterEqual(value, 0)
                self.assertLessEqual(value, 1)


    def test_sample_num(self):
        """
        Test the function of ClsAnalysis() with different sample numbers.
        """
        np.random.seed(123)
        train_features = pd.DataFrame(np.random.rand(14, 10), columns=[f'feature_{i}' for i in range(10)])
        train_labels = pd.Series([0, 1] * 7, name='label')
        train_data = DataBase(features=[train_features], labels=train_labels)

        cls_analysis = ClsAnalysis(
            result_dir="test_dir",
            random_state=123,
            n_feature_to_select=10,
            specified_models=["LR"],
            hpo_algorithm=HPOAlgorithm.RandomSearch,
            hpo_search_iter=1,
            n_bootstrap=10,
            n_jobs=10
        )
        # Fit the model
        cls_analysis.fit(train_data)

        # Transform test data
        test_features = pd.DataFrame(np.random.rand(2, 10), columns=[f'feature_{i}' for i in range(10)])
        test_labels = pd.Series([0, 1], name='label')
        test_data = DataBase(features=[test_features], labels=test_labels)
        cls_analysis.transform(test_data)

        with self.assertRaises(ValueError):
            test_features = pd.DataFrame(np.random.rand(1, 10), columns=[f'feature_{i}' for i in range(10)])
            test_labels = pd.Series([1], name='label')
            test_data = DataBase(features=[test_features], labels=test_labels)
            cls_analysis.transform(test_data)

        train_features = pd.DataFrame(np.random.rand(12, 10), columns=[f'feature_{i}' for i in range(10)])
        train_labels = pd.Series([0, 1] * 6, name='label')
        train_data = DataBase(features=[train_features], labels=train_labels)
        cls_analysis = ClsAnalysis(
            result_dir="test_dir",
            random_state=123,
            n_feature_to_select=10,
            specified_models=["LR"],
            hpo_algorithm=HPOAlgorithm.RandomSearch,
            hpo_search_iter=1,
            n_bootstrap=10,
            n_jobs=10
        )
        # Fit the model
        with self.assertRaises(ValueError):
            cls_analysis.fit(train_data)


if __name__ == "__main__":
    unittest.main()


def main() -> None:
    unittest.main()


if __name__ == "__main__":
    main()