#!/usr/bin/env python3
###############################################################
## Copyright: PromptBio Corp 2022
# Author: jiayuchen
# Date of creation: 07/29/2024
# Date of revision: 08/06/2024
#
## AutoML
## Description: Unit test for MLGenie models
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
from sklearn.datasets import load_iris
from pandas.testing import assert_frame_equal

cur_folder = os.path.dirname(os.path.realpath(__file__))
last_folder = os.path.dirname(cur_folder)
last_last_folder = os.path.dirname(last_folder)
last_last_last_folder = os.path.dirname(last_last_folder)
last_last_last_last_folder = os.path.dirname(last_last_last_folder)
sys.path.append(last_last_last_last_folder)


from MLGenie.design.model.eda import PCA, HC, FA


class TestUnsupervisedModel(unittest.TestCase):

    def setUp(self):
        self.model_list = [PCA, HC, FA]
        self.X = pd.DataFrame(load_iris().data)

    def test_fit(self) -> None:
        for model_name in self.model_list:
            model = model_name()
            model.fit(self.X)

    def test_save(self):
        """Test save function"""
        for model_name in self.model_list:
            # FA model dont need to test save and load
            if model_name.model_type == "FA":
                continue
            path = "./result/{}.pkl".format(model_name.model_type)
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            model = model_name()
            model.save(path)
            self.assertTrue(os.path.exists(path))

            path = "./result/{}.pkl".format(model_name.model_type)
            model = model_name(path_to_store_model=path)
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
            # FA model dont need to test save and load
            if model_name.model_type == "FA":
                continue
            path = "./result/{}.pkl".format(model_name.model_type)
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))

            model = model_name()
            model.fit_transform(self.X)
            model.save(path)

            model2 = model_name()
            model2.load(path)

            result_list1 = model.fit_transform(self.X)
            result_list2 = model2.fit_transform(self.X)
            for res1, res2 in zip(result_list1, result_list2):
                assert_frame_equal(res1, res2)

            # Test invalid path
            with self.assertRaises(ValueError):
                model.load(path=1)

            with self.assertRaises(ValueError):
                model.load(path="./not_exist_path/svm.pkl")


class TestPCA(unittest.TestCase):
    """Test PCA class"""

    def setUp(self):
        self.n_components = 2
        self.random_state = 1234
        self.iris = load_iris()
        self.X = pd.DataFrame(self.iris.data)
        self.model = PCA(n_components=self.n_components, random_state=self.random_state)
        self.pca_matrix, self.pca_variance_ratio = self.model.fit_transform(self.X)

    def test_pca_result(self):
        """Test whether the pca result is correct"""
        x = self.X.copy(deep=True)
        n_components = 2

        ##### Start PCA ####
        # 1. Data decentralization and Calculate the autocovariance matrix
        x -= x.mean(axis=0)
        x_T = np.transpose(x)
        x_cov = np.cov(x_T)

        # 2. Feature decomposition
        eigenvalues, eigenvector = scipy.linalg.eig(x_cov)

        # 3. Eigenvalue sorting
        new_index = np.argsort(eigenvalues)[::-1]
        order_eigenvector = -eigenvector[:, new_index]

        # 4. Calculate transformed features
        transformed_matrix = x.dot(order_eigenvector)
        transformed_x = np.around(transformed_matrix, 3).iloc[:, : self.n_components]
        # Calculate the contribution rates corresponding to all features
        variance_ratio_true = np.around(eigenvalues[new_index] / eigenvalues.sum(), 3)[
            : self.n_components
        ]
        ##### End PCA ####

        # Test whether the result is correct
        assert np.array_equal(
            self.pca_variance_ratio.round(3).values.reshape(self.n_components),
            variance_ratio_true,
        )
        assert np.array_equal(self.pca_matrix.round(3).abs().values, abs(transformed_x))

    def test_parameters(self):
        # Test invalid n_components
        with self.assertRaises(AssertionError):
            model = PCA(n_components=0)

        # Test invalid n_components
        n_samples, n_features = self.X.shape
        with self.assertRaises(AssertionError):
            model = PCA(n_components=min(n_samples, n_features) + 1)
            model.fit(self.X)

        with self.assertRaises(AssertionError):
            model = PCA(n_components=-1)
            model.fit(self.X)

        # Test invalid X
        # 1. empty X
        with self.assertRaises(AssertionError):
            X = pd.DataFrame()
            model = PCA()
            model.fit(X)

        # 2. lack of features
        with self.assertRaises(AssertionError):
            X = self.X.copy(deep=True).iloc[:, :3]
            pca_matrix, _ = self.model.transform(X)

        # 3. add new feature
        with self.assertRaises(AssertionError):
            X = self.X.copy(deep=True)
            X[len(X.columns)] = X[0]
            pca_matrix, _ = self.model.transform(X)

        # 4. features num less than n_components
        with self.assertRaises(AssertionError):
            X = self.X.copy(deep=True)[0]
            self.model.fit_transform(X)

    def test_reproducibility(self):
        model = PCA(n_components=self.n_components, random_state=self.random_state)
        pca_matrix, pca_variance_ratio = model.fit_transform(self.X)
        assert_frame_equal(pca_matrix, self.pca_matrix)
        assert_frame_equal(pca_variance_ratio, self.pca_variance_ratio)


class TestHC(unittest.TestCase):
    """Test Hierarchical Clustering class"""

    def setUp(self):
        self.clusters_limit = (2, 8)
        self.n_clusters = 2
        self.iris = load_iris()
        self.X = pd.DataFrame(self.iris.data)
        self.model = HC(clusters_limit=self.clusters_limit, n_clusters=self.n_clusters)
        self.cluster_labels, self.cluster_tree_structure = self.model.fit_transform(
            self.X
        )

    def test_hc_result(self):
        """Test whether the HC result is correct"""
        # Add test case with calculation result
        data = [
            [1, 2, 3, 4, 5],
            [1, 3, 5, 7, 9],
            [2, 4, 6, 8, 10],
            [7, 5, 3, 1, -1],
            [5, 7, 3, 6, 2],
        ]
        # Start clustering
        data = np.array(data)
        n_samples, n_features = data.shape
        # Caculate distance matrix
        distance_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i):
                dist = np.sqrt(np.sum(np.square(data[i] - data[j])))
                distance_matrix[i][j] = dist
                distance_matrix[j][i] = dist

        tree_structure_true = [[1, 2], [3, 4], [0, 5], [6, 7]]
        cluster_labels_true = [0, 0, 0, 1, 1]

        model = HC(clusters_limit=None, n_clusters=2)
        cluster_labels, cluster_tree_structure = model.fit_transform(pd.DataFrame(data))
        assert np.array_equal(
            cluster_labels.values.reshape(n_samples), np.array(cluster_labels_true)
        )
        assert np.array_equal(
            cluster_tree_structure.values, np.array(tree_structure_true)
        )

    def test_parameters(self):
        # Test invalid clusters_limit
        with self.assertRaises(AssertionError):
            model = HC(clusters_limit=[])

        with self.assertRaises(AssertionError):
            model = HC(clusters_limit={})

        with self.assertRaises(AssertionError):
            model = HC(clusters_limit=(1, 4))

        with self.assertRaises(AssertionError):
            model = HC(clusters_limit=(4, 4))

        with self.assertRaises(AssertionError):
            model = HC(clusters_limit=(5, 4))

        # Test invalid n_clusters
        with self.assertRaises(AssertionError):
            model = HC(n_clusters=[])

        with self.assertRaises(AssertionError):
            model = HC(n_clusters="1")

        with self.assertRaises(AssertionError):
            model = HC(n_clusters=1)

        # Test invalid X
        # 1. num_samples larger than n_clusters
        with self.assertRaises(AssertionError):
            model = HC(clusters_limit=None, n_clusters=len(self.X) + 1)
            model.fit_transform(self.X)

        # 2. empty input matrix
        with self.assertRaises(AssertionError):
            model = HC(clusters_limit=None, n_clusters=len(self.X) + 1)
            X = pd.DataFrame()
            model.fit_transform(X)

        # 3. Only one features
        with self.assertRaises(AssertionError):
            X = self.X[0]
            self.model.fit_transform(X)

    def test_select_best_n_clusters(self):
        """Test _select_best_n_clusters function"""
        model = HC(clusters_limit=(len(self.X) + 1, len(self.X) + 10))
        best_n_clusters = model._select_best_n_clusters(self.X)
        self.assertTrue(best_n_clusters is None)

        with self.assertRaises(ValueError):
            model.fit_transform(self.X)

    def test_reproducibility(self):
        model = HC(clusters_limit=self.clusters_limit, n_clusters=self.n_clusters)
        cluster_labels, cluster_tree_structure = self.model.fit_transform(self.X)
        assert_frame_equal(cluster_labels, self.cluster_labels)
        assert_frame_equal(cluster_tree_structure, self.cluster_tree_structure)


class TestFA(unittest.TestCase):
    """Test Feature Analysis class"""

    def setUp(self):
        self.iris = load_iris()
        self.X = pd.DataFrame(self.iris.data)

        self.top_k = 3
        self.model = FA(top_k=self.top_k)
        self.feature_corr_coef = self.model.fit_transform(self.X)

    def test_fa_result(self):
        """Test whether the HC result is correct"""
        data = [
            [1, 2, 3, 4, 5],
            [1, 3, 5, 7, 9],
            [2, 4, 6, 8, 10],
            [7, 5, 3, 1, -1],
            [5, 7, 3, 6, 2],
            [1, 1, 2, 2, 3],
        ]

        feature_matrix = pd.DataFrame(data, columns=["a", "b", "c", "d", "e"])
        top_k = 10

        # Step1: Test variance
        data = np.array(data)
        var_true = np.var(data, axis=0, ddof=1)
        var_in_workflow = feature_matrix.var()
        assert np.array_equal(var_in_workflow.values, var_true)

        # Step2: Test reorder data according to variance
        sort_index = np.argsort(var_true)[::-1]
        data = data[:, sort_index]

        top_k_index = var_in_workflow.sort_values(ascending=False)[:top_k].index
        filtered_feature_matrix = feature_matrix[top_k_index]
        assert np.array_equal(filtered_feature_matrix.values, data)

        # Step3: Test  Pearson correlation coefficients
        coef = np.around(np.corrcoef(data.T), 3)
        feature_corr_coef = (
            filtered_feature_matrix.corr(method="pearson").fillna(0).round(3)
        )
        assert np.array_equal(feature_corr_coef.values, coef)

        # Step4:Test heatmap columns order
        feature_corr_coef_by_model = FA().fit_transform(feature_matrix)
        columns = feature_corr_coef_by_model.columns.values
        assert np.array_equal(columns, np.array(["e", "d", "a", "b", "c"]))

    def test_parameters(self):
        # Test invalid top_k
        with self.assertRaises(AssertionError):
            model = FA(top_k=0)

        with self.assertRaises(AssertionError):
            model = FA(top_k=None)

        with self.assertRaises(AssertionError):
            model = FA(top_k=[])

        # Test invalid X: empty input matrix
        with self.assertRaises(AssertionError):
            model = HC(clusters_limit=None, n_clusters=len(self.X) + 1)
            X = pd.DataFrame()
            model.fit_transform(X)

    def test_reproducibility(self):
        model = FA(top_k=self.top_k)
        feature_corr_coef = model.fit_transform(self.X)
        assert_frame_equal(feature_corr_coef, self.feature_corr_coef)


if __name__ == "__main__":
    unittest.main()


def main() -> None:
    unittest.main()


if __name__ == "__main__":
    main()
