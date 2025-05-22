#!/usr/bin/env python3
###############################################################
## Copyright: PromptBio Corp 2024
# Author: jiayuchen
# Date of creation: 09/29/2024
# Date of revision: 
#
## MLGenie
## Description: Unit test for AnalysisBase
#
###############################################################

import unittest
import os
import sys
from typing import List, Union, Tuple


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

from MLGenie.design.analyses._base import AnalysisBase
from MLGenie.design.analyses.supervised_analyses import SupervisedAnalysis
from MLGenie.design.Data import MultiOmicsData, BinaryData, ContinousData, DataBase



class TestAnalysisBase(unittest.TestCase):
    """ 
    Test the AnalysisBase class
    """
    def test_init(self):
        # Test valid initialization
        analysis = AnalysisBase(result_dir="test_dir", random_state=42)
        self.assertEqual(analysis.result_dir, "test_dir")
        self.assertEqual(analysis.random_state, 42)

        # Test default random_state
        analysis = AnalysisBase(result_dir="test_dir")
        self.assertEqual(analysis.random_state, 123)

        # Test invalid result_dir type
        with self.assertRaises(TypeError):
            AnalysisBase(result_dir=123)

        # Test invalid random_state type
        with self.assertRaises(TypeError):
            AnalysisBase(random_state="42")

        # Test negative random_state
        with self.assertRaises(ValueError):
            AnalysisBase(random_state=-1)

        # Test directory creation
        test_dir = "temp_test_dir"
        AnalysisBase(result_dir=test_dir)
        self.assertTrue(os.path.exists(test_dir))
        os.rmdir(test_dir)  # Clean up

        # Test random seed setting
        analysis = AnalysisBase(random_state=42)
        np.random.seed(42)
        expected_random = np.random.rand()
        np.random.seed(analysis.random_state)
        actual_random = np.random.rand()
        self.assertEqual(expected_random, actual_random)
        
    def test_preprocess(self):
        analysis = AnalysisBase(result_dir="test_dir")
        
        # Test with ContinousData
        X  = pd.DataFrame(np.random.rand(100, 5))
        y = pd.Series(np.random.randint(0, 2, 100))
        data = ContinousData(X, y)

        # Test the preprocess method
        processed_data = analysis._preprocess(data)

        # Assert that the preprocessed dataframe is not None
        self.assertIsNotNone(processed_data)

        # Assert that the preprocessed dataframe is a ContinousData
        self.assertIsInstance(processed_data, ContinousData)

        # Assert that the preprocessed dataframe has the same number of rows as the input
        self.assertEqual(len(processed_data), len(data))

        # Test with BinaryData
        X_binary = pd.DataFrame(np.random.randint(0, 2, size=(100, 5)))
        y_binary = pd.Series(np.random.randint(0, 2, 100))
        binary_data = BinaryData(X_binary, y_binary)

        # Test the preprocess method with BinaryData
        processed_binary_data = analysis._preprocess(binary_data)

        # Assert that the preprocessed dataframe is not None
        self.assertIsNotNone(processed_binary_data)

        # Assert that the preprocessed dataframe is a BinaryData
        self.assertIsInstance(processed_binary_data, BinaryData)

        # Assert that the preprocessed dataframe has the same number of rows as the input
        self.assertEqual(len(processed_binary_data), len(binary_data))

        # Assert that the processed flag is updated
        self.assertTrue(processed_binary_data.if_processed)

        # Assert that all values in the processed data are still binary (0 or 1)
        self.assertTrue(np.all(np.isin(processed_binary_data.get_features("raw"), [0, 1])))
        self.assertTrue(np.all(np.isin(processed_binary_data.get_features("processed"), [0, 1])))


        

     
if __name__ == "__main__":
    unittest.main()


def main() -> None:
    unittest.main()


if __name__ == "__main__":
    main()
