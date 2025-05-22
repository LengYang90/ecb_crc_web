#!/usr/bin/env python3
###############################################################
## Copyright: PromptBio Corp 2023
# Author: jiayu
# Date of creation: 10/11/2024
# Date of creation: 11/30/2024
#
## MLGenie
## Description: Unit test for utility functions
#
###############################################################
import os
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn import datasets

cur_folder = os.path.dirname(os.path.realpath(__file__))
last_folder = os.path.dirname(cur_folder)
last_last_folder = os.path.dirname(last_folder)

sys.path.append(last_last_folder)

from MLGenie.utils import (
    Metrics,
    Average,
    check_X_y,
    get_scoring_str,
    convert_survival_label,
    prepare_train_test_data
)
from MLGenie.Data import DataBase

class TestUtils(unittest.TestCase):
    """Test utility functions"""

    def setUp(self) -> None:
        # Load the iris dataset.
        iris = datasets.load_iris()
        self.X = pd.DataFrame(iris.data)
        self.X.columns = ["f0", "f1", "f2", "f3"]
        self.y = iris.target

    def test_check_X_y(self) -> None:
        """
        Test check_X_y().
        """
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = np.array([1, 2, 3])
        check_X_y(X, y)
        with self.assertRaises(TypeError):
            check_X_y(X.values, y)
        with self.assertRaises(TypeError):
            check_X_y(X, y.tolist())

    def test_get_scoring_str(self) -> None:
        """
        Test get_scoring_str().
        """
        self.assertEqual(get_scoring_str(Metrics.Accuracy, None), "accuracy")
        self.assertEqual(get_scoring_str(Metrics.Fscore, Average.Binary), "f1")
        self.assertEqual(get_scoring_str(Metrics.Fscore, Average.Micro), "f1_micro")
        self.assertEqual(get_scoring_str(Metrics.Fscore, Average.Macro), "f1_macro")
        self.assertEqual(
            get_scoring_str(Metrics.Fscore, Average.Weighted), "f1_weighted"
        )
        self.assertEqual(
            get_scoring_str(Metrics.Precision, Average.Binary), "precision"
        )
        self.assertEqual(
            get_scoring_str(Metrics.Precision, Average.Micro), "precision_micro"
        )
        self.assertEqual(
            get_scoring_str(Metrics.Precision, Average.Macro), "precision_macro"
        )
        self.assertEqual(
            get_scoring_str(Metrics.Precision, Average.Weighted), "precision_weighted"
        )
        self.assertEqual(get_scoring_str(Metrics.Recall, Average.Binary), "recall")
        self.assertEqual(get_scoring_str(Metrics.Recall, Average.Micro), "recall_micro")
        self.assertEqual(get_scoring_str(Metrics.Recall, Average.Macro), "recall_macro")
        self.assertEqual(
            get_scoring_str(Metrics.Recall, Average.Weighted), "recall_weighted"
        )
        self.assertEqual(get_scoring_str(Metrics.AUC, None), "roc_auc")
        self.assertEqual(
            get_scoring_str(Metrics.CorrCoef, None), "correlation_coefficient"
        )
        self.assertEqual(get_scoring_str(Metrics.MAE, None), "neg_mean_absolute_error")
        self.assertEqual(get_scoring_str(Metrics.MSE, None), "neg_mean_squared_error")
        self.assertEqual(get_scoring_str(Metrics.R2, None), "r2")

    def test_convert_survival_label(self) -> None:
        """
        Test convert_survival_label() function.
        """
        labels = pd.DataFrame(
            {
                "event": [True, False, True, False],
                "time": [1, 2, 3, 4],
                "other": [1, 2, 3, 4],
            }
        )
        y = convert_survival_label(labels)
        self.assertEqual(y.shape, (4,))
        assert np.equal(y[0][0], True)
        assert np.equal(y[1][0], False)
        assert np.equal(y[2][0], True)
        assert np.equal(y[3][0], False)

        assert np.equal(y[0][1], float(1))
        assert np.equal(y[1][1], float(2))
        assert np.equal(y[2][1], float(3))
        assert np.equal(y[3][1], float(4))

        with self.assertRaises(TypeError):
            convert_survival_label([1, 2, 3, 4])
        with self.assertRaises(ValueError):
            convert_survival_label(pd.DataFrame({"event": [True, False]}))
        with self.assertRaises(ValueError):
            convert_survival_label(pd.DataFrame({"time": [1, 2]}))

    def test_prepare_train_test_data(self) -> None:
        """
        Test prepare_train_test_data() function.
        """
        

        # Create a sample data with shape (100,10)
        X = pd.DataFrame(np.random.rand(100, 10))
        y = pd.Series(np.random.rand(100))
        data = DataBase(features=X, labels=y)

        # Test with valid parameters
        train_data, test_data = prepare_train_test_data(data, 0.2, 123)
        self.assertEqual(train_data.get_shape(), (80, 10))  # 80% of 100 for training
        self.assertEqual(train_data.get_labels().shape, (80,))
        self.assertEqual(test_data.get_shape(), (20, 10))  # 20% of 100 for testing
        self.assertEqual(test_data.get_labels().shape, (20,))


        with self.assertRaises(ValueError):
            prepare_train_test_data(None, 0.2, 123)
        with self.assertRaises(ValueError):
            prepare_train_test_data(data, -0.2, 123)
        with self.assertRaises(ValueError):
            prepare_train_test_data(data, 0.2, "123")

def main() -> None:
    unittest.main()


if __name__ == "__main__":
    main()
