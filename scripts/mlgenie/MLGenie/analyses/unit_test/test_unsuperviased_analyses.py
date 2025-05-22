#!/usr/bin/env python3
###############################################################
## Copyright: PromptBio Corp 2024
# Author: jiayuchen
# Date of creation: 10/09/2024
# Date of revision: 
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

cur_folder = os.path.dirname(os.path.realpath(__file__))
last_folder = os.path.dirname(cur_folder)
last_last_folder = os.path.dirname(last_folder)
last_last_last_folder = os.path.dirname(last_last_folder)
last_last_last_last_folder = os.path.dirname(last_last_last_folder)
sys.path.append(last_last_folder)
sys.path.append(last_last_last_folder)
sys.path.append(last_last_last_last_folder)

from MLGenie.design.analyses.unsupervised_analyses import UnsupervisedAnalysis
from MLGenie.design.Data import MultiOmicsData,DataBase, ContinousData,BinaryData
from MLGenie.design.model.eda import PCA, HC, FA

class TestUnsupervisedAnalysis(unittest.TestCase):
    """ 
    Test the UnsupervisedAnalysis class
    """
    def test_init(self):
        # Test valid initialization
        analysis = UnsupervisedAnalysis(result_dir="test_dir", random_state=42)
        self.assertEqual(analysis.result_dir, "test_dir")
        self.assertEqual(analysis.random_state, 42)

        # Test invalid result_dir type
        with self.assertRaises(TypeError):
            UnsupervisedAnalysis(result_dir=123)

        # Test invalid random_state type
        with self.assertRaises(TypeError):
            UnsupervisedAnalysis(random_state="42")

        # Test invalid n_components type
        with self.assertRaises(ValueError):
            UnsupervisedAnalysis(n_components="2")

        # Test invalid dimensionality_reduction type
        with self.assertRaises(ValueError):
            UnsupervisedAnalysis(dimensionality_reduction=123)

        # Test invalid top_k type
        with self.assertRaises(ValueError):
            UnsupervisedAnalysis(top_k="1000")
        
    def test_check_n_components(self):
        # Test with n_components less than or equal to the number of features
        analysis = UnsupervisedAnalysis(n_components=2)
        data = DataBase(features=[pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_{i}' for i in range(10)])], labels=pd.Series(np.random.randint(0, 2, 100), name='label'))
        analysis._check_n_components(data)
        self.assertEqual(analysis.n_components, 2)

        # Test with n_components equal to the number of features
        analysis = UnsupervisedAnalysis(n_components=10)
        data = DataBase(features=[pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_{i}' for i in range(10)])], labels=pd.Series(np.random.randint(0, 2, 100), name='label'))
        analysis._check_n_components(data)
        self.assertEqual(analysis.n_components, 10)

        # Test with n_components greater than the number of features
        analysis = UnsupervisedAnalysis(n_components=11)
        data = DataBase(features=[pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_{i}' for i in range(10)])], labels=pd.Series(np.random.randint(0, 2, 100), name='label'))
        analysis._check_n_components(data)
        self.assertEqual(analysis.n_components, 10)

    def test_fit(self):
        # Test with normal data input
        analysis = UnsupervisedAnalysis()
        features = pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_{i}' for i in range(10)])
        labels = pd.Series(np.random.randint(0, 2, 100), name='label')
        data = DataBase(features=features, labels=labels)
        analysis.fit(data)

        # Test with binary data input
        analysis = UnsupervisedAnalysis()
        features = pd.DataFrame(np.random.randint(0, 2, (100, 100)), columns=[f'feature_{i}' for i in range(100)])
        labels = pd.Series(np.random.randint(0, 2, 100), name='label')
        binary_data = BinaryData(features=features, labels=labels)
        analysis.fit(binary_data)

        # Test with continuous data input
        analysis = UnsupervisedAnalysis()
        features = pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_{i}' for i in range(10)])
        labels = pd.Series(np.random.randint(0, 2, 100), name='label')
        data = ContinousData(features=features, labels=labels)
        analysis.fit(data)
    
    def test_transform(self):
        # Test with normal data input
        n_components = 5
        top_k = 1000
        n_clusters = 2
        n_samples = 100
        n_features = 10

        analysis = UnsupervisedAnalysis(n_components=n_components, top_k=top_k, n_clusters=n_clusters)
        features = pd.DataFrame(np.random.rand(n_samples, n_features), columns=[f'feature_{i}' for i in range(10)])
        labels = pd.Series(np.random.randint(0, 2, n_samples), name='label')
        data = DataBase(features=features, labels=labels)
        analysis.fit(data)
        result = analysis.transform(data)
        # for k,v in result.items():
        #     print(k,v.shape)
        # Check if the result is a dictionary
        self.assertIsInstance(result, dict)
        
        # Check if all expected keys are present in the result
        expected_keys = ["pca_matrix", "pca_variance_ratio", "cluster_labels", "cluster_tree_structure", "feature_corr_coef", "feature_statistics", "processed_feature_matrix"]
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check if the values of the expected keys are not None
        for key in expected_keys:
            self.assertIsNotNone(result[key])

        # Check the shape and type of each item in the result
        expected_shapes = {
            "pca_matrix": (n_samples, n_components),  
            "pca_variance_ratio": (n_components,1),  
            "cluster_labels": (n_samples,1),  
            "cluster_tree_structure": (n_samples-1,2),  
            "feature_corr_coef": (n_features, n_features),
            "feature_statistics": (n_features, 12)
        }
        expected_types = {
            "pca_matrix": pd.DataFrame,
            "pca_variance_ratio": pd.DataFrame,
            "cluster_labels": pd.DataFrame,
            "cluster_tree_structure": pd.DataFrame,
            "feature_corr_coef": pd.DataFrame,
            "feature_statistics": pd.DataFrame
        }

        for key, value in result.items():
            if key == "processed_feature_matrix":
                continue
            self.assertEqual(value.shape, expected_shapes[key], f"Shape of {key} is incorrect")
            self.assertIsInstance(value, expected_types[key], f"Type of {key} is incorrect")

        result2 = analysis.fit_transform(data)
        # Check if the result is a dictionary
        self.assertIsInstance(result2, dict)
        
        # Check if all expected keys are present in the result
        expected_keys = ["pca_matrix", "pca_variance_ratio", "cluster_labels", "cluster_tree_structure", "feature_corr_coef", "feature_statistics"]
        for key in expected_keys:
            self.assertIn(key, result2)
        
        for key in expected_keys:
            self.assertIsNotNone(result2[key])
        
    def test_save_result(self):
        analysis = UnsupervisedAnalysis()
        features = pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_{i}' for i in range(10)])
        labels = pd.Series(np.random.randint(0, 2, 100), name='label')
        data = DataBase(features=features, labels=labels)
        
        analysis = UnsupervisedAnalysis(result_dir="test_dir")
        analysis.fit(data)
        result = analysis.transform(data)
        analysis.save_result(result)
        # Check if the files are saved correctly
        expected_files = ["processed_feature_matrix.csv", "feature_statistics.csv", "pca_matrix.csv", "pca_variance_ratio.csv", "cluster_labels.csv", "cluster_tree_structure.csv", "feature_corr_coef.csv"]
        for file in expected_files:
            self.assertTrue(os.path.exists(os.path.join(analysis.result_dir, file)), f"{file} is not saved correctly")

if __name__ == "__main__":
    unittest.main()


def main() -> None:
    unittest.main()


if __name__ == "__main__":
    main()
