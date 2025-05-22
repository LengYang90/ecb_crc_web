#!/usr/bin/env python3
###############################################################
## Copyright: PromptBio Corp 2022
# Author: jiayuchen
# Date of creation: 08/29/2024
# Date of revision:
#
## AutoML
## Description: Unit test for MLGenie regression models
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
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

cur_folder = os.path.dirname(os.path.realpath(__file__))
last_folder = os.path.dirname(cur_folder)
last_last_folder = os.path.dirname(last_folder)
last_last_last_folder = os.path.dirname(last_last_folder)
last_last_last_last_folder = os.path.dirname(last_last_last_folder)
sys.path.append(last_last_last_last_folder)


from MLGenie.design.model.regression import LassoR, RidgeR, ElasticR, GBDTR, RFR, DTR
from MLGenie.design.utils import (
    HPOAlgorithm,
    Metrics,
    Average,
)
from MLGenie.design.hpo_param_config import RegressionParamConfig


class TestRegressionModel(unittest.TestCase):
    """Test RegressionModel class"""

    def setUp(self) -> None:
        self.model_list = [
            LassoR,
            RidgeR,
            ElasticR,
            # GBDTR,
            # RFR,
            # DTR
        ]
        X, y = make_regression(n_samples=1000, n_features=4)
        y = y - np.min(y) + 1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        self.X = pd.DataFrame(X_train)
        self.y = pd.Series(y_train)
        self.test_x = pd.DataFrame(X_test)
        self.test_y = pd.Series(y_test)
        self.fit_performance_threshold = 0.1

    def test_init(self) -> None:
        """Test init function"""
        for model_name in self.model_list:

            # Test invalid hpo_algorithm value
            with self.assertRaises(AssertionError):
                svm_model = model_name(hpo_algorithm="hpo_algorithm")

            with self.assertRaises(AssertionError):
                svm_model = model_name(hpo_algorithm=None)

            with self.assertRaises(AssertionError):
                svm_model = model_name(hpo_algorithm=HPOAlgorithm)

            # Test invalid hpo_search_iter value
            with self.assertRaises(AssertionError):
                svm_model = model_name(hpo_search_iter=0)

            with self.assertRaises(AssertionError):
                svm_model = model_name(hpo_search_iter=-100)

            with self.assertRaises(AssertionError):
                svm_model = model_name(hpo_search_iter=None)

            # Test invalid cv value
            with self.assertRaises(AssertionError):
                svm_model = model_name(cv=0)

            with self.assertRaises(AssertionError):
                svm_model = model_name(cv=-1)

            with self.assertRaises(AssertionError):
                svm_model = model_name(cv=None)

            # Test invalid random_state value
            with self.assertRaises(AssertionError):
                svm_model = model_name(random_state=-100)

            with self.assertRaises(AssertionError):
                svm_model = model_name(random_state=None)

            # Test invalid n_jobs value
            with self.assertRaises(AssertionError):
                svm_model = model_name(n_jobs=0)

            with self.assertRaises(AssertionError):
                svm_model = model_name(n_jobs=-1)

            with self.assertRaises(AssertionError):
                svm_model = model_name(n_jobs=None)

            # Test invalid path_to_store_model value
            with self.assertRaises(TypeError):
                svm_model = model_name(path_to_store_model=1)

            with self.assertRaises(FileNotFoundError):
                svm_model = model_name(path_to_store_model="model.pkl")

    def test_get_hyper_params_space(self) -> None:
        """Test get_hyper_params_space function"""
        for model_name in self.model_list:
            for hpo_algorithm_config in HPOAlgorithm:
                model = model_name(hpo_algorithm=hpo_algorithm_config)
                param_config = model._get_hyper_params_space()
                self.assertFalse(param_config is None)

    def test_HPO(self) -> None:
        """Test HPO function"""
        for model_name in self.model_list:
            model = model_name(
                hpo_algorithm=HPOAlgorithm.GridSearch, hpo_search_iter=100, n_jobs=8
            )
            best_params, best_score, cv_results = model._HPO(self.X, self.y)
            self.assertTrue(isinstance(best_params, dict))
            self.assertTrue(isinstance(best_score, float))
            self.assertTrue(isinstance(cv_results, dict))

            model = model_name(
                hpo_algorithm=HPOAlgorithm.RandomSearch, hpo_search_iter=10
            )
            best_params, best_score, cv_results = model._HPO(self.X, self.y)
            self.assertTrue(isinstance(best_params, dict))
            self.assertTrue(isinstance(best_score, float))
            self.assertTrue(isinstance(cv_results, dict))

            model = model_name(
                hpo_algorithm=HPOAlgorithm.BayesianSearch, hpo_search_iter=20
            )
            best_params, best_score, cv_results = model._HPO(self.X, self.y)
            self.assertTrue(isinstance(best_params, dict))
            self.assertTrue(isinstance(best_score, float))
            self.assertTrue(isinstance(cv_results, dict))

    def test_fit(self) -> None:
        """Test fit function"""
        for model_name in self.model_list:
            print(
                "\n\n======in {} model fit function=======\n".format(
                    model_name.model_type
                )
            )
            # Get default_performance without HPO
            default_model = model_name()
            default_model.model.fit(self.X, self.y)
            default_model.n_features = len(self.X.columns)
            default_performance = default_model.evaluate(self.test_x, self.test_y)
            print("default_performance=", default_performance)

            model = model_name(
                hpo_algorithm=HPOAlgorithm.GridSearch,
                hpo_search_iter=50,
                cv=5,
                random_state=123,
                n_jobs=8,
            )

            model.fit(self.X, self.y)
            performance = model.evaluate(self.test_x, self.test_y)
            print("GridSearch performance=", performance)
            #  Test if the model is fitted actually
            # self.assertGreater(performance, self.fit_performance_threshold)

            model = model_name(
                hpo_algorithm=HPOAlgorithm.RandomSearch,
                hpo_search_iter=1,
                cv=4,
                random_state=123,
                n_jobs=3,
            )
            model.fit(self.X, self.y)
            performance = model.evaluate(self.test_x, self.test_y)
            print("RandomSearch performance=", performance)
            #  Test if the model is fitted actually
            # self.assertGreater(performance, self.fit_performance_threshold)

            model = model_name(
                hpo_algorithm=HPOAlgorithm.BayesianSearch,
                hpo_search_iter=1,
                cv=4,
                random_state=222,
                n_jobs=8,
            )
            model.fit(self.X, self.y)
            performance = model.evaluate(self.test_x, self.test_y)
            print("BayesianSearch performance=", performance)
            #  Test if the model is fitted actually
            # self.assertGreater(performance, self.fit_performance_threshold)

            # Test invalid x, y
            with self.assertRaises(ValueError):
                x = self.X.copy(deep=True)[5:]
                model = model_name()
                model.fit(x, self.y)

            with self.assertRaises(ValueError):
                y = self.y.copy(deep=True)[5:]
                model = model_name()
                model.fit(self.X, y)

    def test_predict(self) -> None:
        """Test predict function"""
        for model_name in self.model_list:
            model = model_name(
                hpo_algorithm=HPOAlgorithm.RandomSearch,
                hpo_search_iter=1,
                cv=5,
                random_state=123,
                n_jobs=8,
            )
            model.fit(self.X, self.y)
            y = model.predict(self.test_x)
            self.assertEqual(len(y), len(self.test_y))

            # Test different X such as the case that lack of features
            with self.assertRaises(AssertionError):
                test_x = self.test_x.copy(deep=True)
                test_x = test_x.drop(test_x.columns[2], axis=1)
                model.predict(test_x)

            # Test different X such as the case that adding new features
            with self.assertRaises(AssertionError):
                test_x = self.test_x.copy(deep=True)
                test_x[len(test_x.columns)] = test_x[0]
                model.predict(test_x)

    def test_evaluate(self) -> None:
        """Test evaluate function"""
        for model_name in self.model_list:
            model = model_name(
                hpo_algorithm=HPOAlgorithm.GridSearch,
                hpo_search_iter=100,
            )
            model.fit(self.X, self.y)
            performance = model.evaluate(self.test_x, self.test_y)
            self.assertTrue(isinstance(performance, float))
            print(f"Performance in evaluate:{performance}")
            # self.assertTrue(performance >= 0 and performance <= 1.0)

    def test_save(self):
        """Test save function"""
        for model_name in self.model_list:
            path = "./result/{}.pkl".format(model_name.model_type)
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            model = model_name(
                hpo_algorithm=HPOAlgorithm.RandomSearch,
                hpo_search_iter=3,
            )
            model.save(path)
            self.assertTrue(os.path.exists(path))

            path = "./result/{}.pkl".format(model_name.model_type)
            model = model_name(hpo_search_iter=1, path_to_store_model=path)
            model.save()
            self.assertTrue(os.path.exists(path))

            with self.assertRaises(ValueError):
                model.save(path=1)

            with self.assertRaises(ValueError):
                model.save(path="./not_exist_path/svm.pkl")

            with self.assertRaises(ValueError):
                model = model_name(path_to_store_model=None)
                model.save()

    def test_load(self):
        """Test load function"""
        for model_name in self.model_list:
            path = "./result/{}.pkl".format(model_name.model_type)
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))

            model = model_name(
                hpo_algorithm=HPOAlgorithm.RandomSearch, hpo_search_iter=10
            )
            model.fit(self.X, self.y)
            model.save(path)

            model2 = model_name()
            model2.load(path)

            performance_true = model.evaluate(self.test_x, self.test_y)
            performance = model2.evaluate(self.test_x, self.test_y)
            self.assertEqual(performance, performance_true)

            # Test invalid path
            with self.assertRaises(ValueError):
                model.load(path=1)

            with self.assertRaises(ValueError):
                model.load(path="./not_exist_path/svm.pkl")

    def test_reproducibility(self):
        """
        check reproducibility of the same random_state
        """
        print("\n===========test_reproducibility=============\n")
        for model_name in self.model_list:
            model1 = model_name(
                hpo_algorithm=HPOAlgorithm.GridSearch,
                hpo_search_iter=10,
                random_state=1234,
            )
            model1.fit(self.X, self.y)
            performance1 = model1.evaluate(self.test_x, self.test_y)

            model2 = model_name(
                hpo_algorithm=HPOAlgorithm.GridSearch,
                hpo_search_iter=10,
                random_state=1234,
            )
            model2.fit(self.X, self.y)
            performance2 = model2.evaluate(self.test_x, self.test_y)
            print(f"performance1:{performance1}")
            print(f"performance2:{performance2}")

            self.assertEqual(performance1, performance2)

    def test_k_fold_cross_validation(self) -> None:
        """
        Test k_fold_cross_validation function
        """
        for model_name in self.model_list:
            model = model_name()
            performance = model.k_fold_cross_validation(self.X, self.y)
            self.assertTrue(performance >= 0)

        for model_name in self.model_list:
            model = model_name()
            with self.assertRaises(TypeError):
                performance = model.k_fold_cross_validation(self.X.values, self.y)

        for model_name in self.model_list:
            model = model_name()
            with self.assertRaises(TypeError):
                performance = model.k_fold_cross_validation(
                    self.X, self.y.values.tolist()
                )


if __name__ == "__main__":
    unittest.main()


def main() -> None:
    unittest.main()


if __name__ == "__main__":
    main()
