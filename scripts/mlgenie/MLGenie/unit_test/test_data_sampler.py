#!/usr/bin/env python3
###############################################################
## Copyright: PromptBio Corp 2022
# Author: jiayuchen
# Date of creation: 09/24/2024
# Date of revision:
#
## AutoML
## Description: Unit test for Data module
#
###############################################################

import unittest
import os
import sys
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal

cur_folder = os.path.dirname(os.path.realpath(__file__))
last_folder = os.path.dirname(cur_folder)
last_last_folder = os.path.dirname(last_folder)

sys.path.append(last_last_folder)

from MLGenie.Data import DataBase, MultiOmicsData, BinaryData, ContinousData
from MLGenie.DataSampler import (
    DataSamplerBase,
    DataSamplerSplit,
    DataSamplerKFoldCV,
    DataSamplerLOOCV,
    DataSamplerBootstrap,
)


class TestDataSamplerSplit(unittest.TestCase):
    """Test DataSamplerSplit class"""

    def setUp(self) -> None:
        self.n_samples = 1000
        self.n_features = 20
        self.X = pd.DataFrame(np.random.rand(self.n_samples, self.n_features))
        self.y = pd.Series(np.random.rand(self.n_samples))
        self.data = DataBase(features=self.X, labels=self.y)
        self.test_size = 0.2
        self.data_sampler1 = DataSamplerSplit(test_size=self.test_size, data=self.data)
        self.data_sampler2 = DataSamplerSplit(
            test_size=self.test_size, X=self.X, y=self.y
        )

    def test_init(self) -> None:
        """Test init function"""
        # Test invalid test_size
        with self.assertRaises(ValueError):
            DataSamplerSplit(test_size=-0.1, data=self.data)
        with self.assertRaises(ValueError):
            DataSamplerSplit(test_size=1.1, data=self.data)
        with self.assertRaises(ValueError):
            DataSamplerSplit(test_size="0.2", data=self.data)

    def test_resplit(self) -> None:
        """Test resplit function"""
        self.data_sampler1.resplit(random_state=42)
        self.data_sampler2.resplit(random_state=42)

        # Check if splits are created
        self.assertIsNotNone(self.data_sampler1.splits)
        self.assertIsNotNone(self.data_sampler2.splits)

        # Check if the split sizes are correct
        train_index, test_index = self.data_sampler1.splits[0]
        self.assertAlmostEqual(
            len(test_index) / self.n_samples, self.test_size, places=2
        )

    def test_len(self) -> None:
        """Test __len__ method"""
        self.assertEqual(len(self.data_sampler1), 1)
        self.assertEqual(len(self.data_sampler2), 1)

    def test_getitem(self) -> None:
        """Test __getitem__ method"""
        # Test valid index
        train_set, test_set = self.data_sampler1[0]
        self.assertIsInstance(train_set, DataBase)
        self.assertIsInstance(test_set, DataBase)

        train_x, test_x, train_y, test_y = self.data_sampler2[0]
        self.assertIsInstance(train_x, pd.DataFrame)
        self.assertIsInstance(test_x, pd.DataFrame)
        self.assertIsInstance(train_y, pd.Series)
        self.assertIsInstance(test_y, pd.Series)

        # Test invalid index
        with self.assertRaises(IndexError):
            self.data_sampler1[1]

    def test_check_data(self) -> None:
        """Test _check_data function"""
        with self.assertRaises(ValueError):
            DataSamplerSplit(test_size=self.test_size, X=self.X)
        with self.assertRaises(ValueError):
            DataSamplerSplit(test_size=self.test_size, y=self.y)
        with self.assertRaises(ValueError):
            DataSamplerSplit(test_size=self.test_size, data=self.data, X=self.X)

        with self.assertRaises(ValueError):
            DataSamplerSplit(test_size=self.test_size)

        with self.assertRaises(ValueError):
            DataSamplerSplit(
                test_size=self.test_size, data=self.data.get_features("raw")
            )

        with self.assertRaises(ValueError):
            DataSamplerSplit(test_size=self.test_size, X=self.X.values, y=self.y)

        with self.assertRaises(ValueError):
            DataSamplerSplit(test_size=self.test_size, X=self.X, y=self.y.values)


class TestDataSamplerKFoldCV(unittest.TestCase):
    """Test DataSamplerKFoldCV class"""

    def setUp(self) -> None:
        self.n_samples = 1000
        self.n_features = 20
        self.X = pd.DataFrame(np.random.rand(self.n_samples, self.n_features))
        self.y = pd.Series(np.random.rand(self.n_samples))
        self.data = DataBase(features=self.X, labels=self.y)
        self.n_splits = 5
        self.data_sampler1 = DataSamplerKFoldCV(n_splits=self.n_splits, data=self.data)
        self.data_sampler2 = DataSamplerKFoldCV(
            n_splits=self.n_splits, X=self.X, y=self.y
        )

    def test_init(self) -> None:
        """Test init function"""
        # Test invalid n_splits
        with self.assertRaises(ValueError):
            DataSamplerKFoldCV(n_splits=1, data=self.data)
        with self.assertRaises(ValueError):
            DataSamplerKFoldCV(n_splits=-1, data=self.data)
        with self.assertRaises(ValueError):
            DataSamplerKFoldCV(n_splits="5", data=self.data)

    def test_resplit(self) -> None:
        """Test resplit function"""
        self.data_sampler1.resplit(random_state=42)
        self.data_sampler2.resplit(random_state=42)

        # Check if splits are created
        self.assertIsNotNone(self.data_sampler1.splits)
        self.assertIsNotNone(self.data_sampler2.splits)

        # Check if the number of splits is correct
        self.assertEqual(len(self.data_sampler1.splits), self.n_splits)
        self.assertEqual(len(self.data_sampler2.splits), self.n_splits)

    def test_len(self) -> None:
        """Test __len__ method"""
        self.assertEqual(len(self.data_sampler1), self.n_splits)
        self.assertEqual(len(self.data_sampler2), self.n_splits)

    def test_getitem(self) -> None:
        """Test __getitem__ method"""
        # Test valid index
        train_set, test_set = self.data_sampler1[0]
        self.assertIsInstance(train_set, DataBase)
        self.assertIsInstance(test_set, DataBase)

        train_x, test_x, train_y, test_y = self.data_sampler2[0]
        self.assertIsInstance(train_x, pd.DataFrame)
        self.assertIsInstance(test_x, pd.DataFrame)
        self.assertIsInstance(train_y, pd.Series)
        self.assertIsInstance(test_y, pd.Series)

        # Test invalid index
        with self.assertRaises(IndexError):
            self.data_sampler1[self.n_splits]

    def test_check_data(self) -> None:
        """Test _check_data function"""
        with self.assertRaises(ValueError):
            DataSamplerKFoldCV(n_splits=self.n_splits, X=self.X)
        with self.assertRaises(ValueError):
            DataSamplerKFoldCV(n_splits=self.n_splits, y=self.y)
        with self.assertRaises(ValueError):
            DataSamplerKFoldCV(n_splits=self.n_splits, data=self.data, X=self.X)
        with self.assertRaises(ValueError):
            DataSamplerKFoldCV(n_splits=self.n_splits)
        with self.assertRaises(ValueError):
            DataSamplerKFoldCV(
                n_splits=self.n_splits, data=self.data.get_features("raw")
            )
        with self.assertRaises(ValueError):
            DataSamplerKFoldCV(n_splits=self.n_splits, X=self.X.values, y=self.y)
        with self.assertRaises(ValueError):
            DataSamplerKFoldCV(n_splits=self.n_splits, X=self.X, y=self.y.values)


class TestDataSamplerLOOCV(unittest.TestCase):
    """Test DataSamplerLOOCV class"""

    def setUp(self) -> None:
        self.n_samples = 100
        self.n_features = 2
        self.X = pd.DataFrame(np.random.rand(self.n_samples, self.n_features))
        self.y = pd.Series(np.random.rand(self.n_samples))
        self.data = DataBase(features=self.X, labels=self.y)
        self.data_sampler1 = DataSamplerLOOCV(data=self.data)
        self.data_sampler2 = DataSamplerLOOCV(X=self.X, y=self.y)

    def test_init(self) -> None:
        """Test initialization"""
        self.assertEqual(len(self.data_sampler1), self.n_samples)
        self.assertEqual(len(self.data_sampler2), self.n_samples)

    def test_resplit(self) -> None:
        """Test resplit function"""
        self.data_sampler1.resplit(random_state=42)
        self.data_sampler2.resplit(random_state=42)

        self.assertIsNotNone(self.data_sampler1.splits)
        self.assertIsNotNone(self.data_sampler2.splits)

        self.assertEqual(len(self.data_sampler1.splits), self.n_samples)
        self.assertEqual(len(self.data_sampler2.splits), self.n_samples)

        # Check that each split leaves out exactly one sample
        for train_idx, test_idx in self.data_sampler1.splits:
            self.assertEqual(len(test_idx), 1)
            self.assertEqual(len(train_idx), self.n_samples - 1)

    def test_len(self) -> None:
        """Test __len__ method"""
        self.assertEqual(len(self.data_sampler1), self.n_samples)
        self.assertEqual(len(self.data_sampler2), self.n_samples)

    def test_getitem(self) -> None:
        """Test __getitem__ method"""
        # Test valid index
        train_set, test_set = self.data_sampler1[0]
        self.assertIsInstance(train_set, DataBase)
        self.assertIsInstance(test_set, DataBase)
        self.assertEqual(len(test_set), 1)
        self.assertEqual(len(train_set), self.n_samples - 1)

        train_x, test_x, train_y, test_y = self.data_sampler2[0]
        self.assertIsInstance(train_x, pd.DataFrame)
        self.assertIsInstance(test_x, pd.DataFrame)
        self.assertIsInstance(train_y, pd.Series)
        self.assertIsInstance(test_y, pd.Series)
        self.assertEqual(len(test_x), 1)
        self.assertEqual(len(train_x), self.n_samples - 1)

        # Test invalid index
        with self.assertRaises(IndexError):
            self.data_sampler1[self.n_samples]

    def test_check_data(self) -> None:
        """Test _check_data function"""
        with self.assertRaises(ValueError):
            DataSamplerLOOCV(X=self.X)
        with self.assertRaises(ValueError):
            DataSamplerLOOCV(y=self.y)
        with self.assertRaises(ValueError):
            DataSamplerLOOCV(data=self.data, X=self.X)
        with self.assertRaises(ValueError):
            DataSamplerLOOCV()
        with self.assertRaises(ValueError):
            DataSamplerLOOCV(data=self.data.get_features("raw"))
        with self.assertRaises(ValueError):
            DataSamplerLOOCV(X=self.X.values, y=self.y)
        with self.assertRaises(ValueError):
            DataSamplerLOOCV(X=self.X, y=self.y.values)


class TestDataSamplerBootstrap(unittest.TestCase):
    """Test DataSamplerBootstrap class"""

    def setUp(self) -> None:
        self.n_samples = 1000
        self.n_features = 5
        self.X = pd.DataFrame(np.random.rand(self.n_samples, self.n_features))
        self.y = pd.Series(np.random.randint(0, 2, self.n_samples))
        self.data = DataBase(features=self.X, labels=self.y)
        self.n_bootstrap = 10
        self.test_size = 0.2
        self.feature_size = 0.8
        self.feature_noise_gaussian = 0.01
        self.feature_noise_bernoulli = 0.01
        self.label_shuffle_size = 0.01
        self.data_sampler1 = DataSamplerBootstrap(
            n_samples=self.n_bootstrap,
            test_size=self.test_size,
            data=self.data,
            feature_size=self.feature_size,
            feature_noise_gaussian=self.feature_noise_gaussian,
            feature_noise_bernoulli=self.feature_noise_bernoulli,
            label_shuffle_size=self.label_shuffle_size,
        )
        self.data_sampler2 = DataSamplerBootstrap(
            n_samples=self.n_bootstrap,
            test_size=self.test_size,
            X=self.X,
            y=self.y,
            feature_size=self.feature_size,
            feature_noise_gaussian=self.feature_noise_gaussian,
            feature_noise_bernoulli=self.feature_noise_bernoulli,
            label_shuffle_size=self.label_shuffle_size,
        )

    def test_init(self) -> None:
        """Test __init__ method"""
        # Test valid initialization
        self.assertEqual(self.data_sampler1.n_samples, self.n_bootstrap)
        self.assertEqual(self.data_sampler1.test_size, self.test_size)
        self.assertEqual(self.data_sampler1.feature_size, self.feature_size)
        self.assertEqual(
            self.data_sampler1.feature_noise_gaussian, self.feature_noise_gaussian
        )
        self.assertEqual(
            self.data_sampler1.feature_noise_bernoulli, self.feature_noise_bernoulli
        )
        self.assertEqual(self.data_sampler1.label_shuffle_size, self.label_shuffle_size)

        # Test invalid n_samples
        with self.assertRaises(ValueError):
            DataSamplerBootstrap(n_samples=0, test_size=self.test_size, data=self.data)
        with self.assertRaises(ValueError):
            DataSamplerBootstrap(n_samples=-1, test_size=self.test_size, data=self.data)
        with self.assertRaises(ValueError):
            DataSamplerBootstrap(
                n_samples=1.5, test_size=self.test_size, data=self.data
            )

        # Test invalid test_size
        with self.assertRaises(ValueError):
            DataSamplerBootstrap(
                n_samples=self.n_bootstrap, test_size=0, data=self.data
            )
        with self.assertRaises(ValueError):
            DataSamplerBootstrap(
                n_samples=self.n_bootstrap, test_size=1, data=self.data
            )
        with self.assertRaises(ValueError):
            DataSamplerBootstrap(
                n_samples=self.n_bootstrap, test_size=-0.1, data=self.data
            )
        with self.assertRaises(ValueError):
            DataSamplerBootstrap(
                n_samples=self.n_bootstrap, test_size=1.1, data=self.data
            )

        # Test invalid feature_size
        with self.assertRaises(ValueError):
            DataSamplerBootstrap(
                n_samples=self.n_bootstrap,
                test_size=self.test_size,
                data=self.data,
                feature_size=0,
            )
        with self.assertRaises(ValueError):
            DataSamplerBootstrap(
                n_samples=self.n_bootstrap,
                test_size=self.test_size,
                data=self.data,
                feature_size=1.1,
            )

        # Test invalid noise parameters
        with self.assertRaises(ValueError):
            DataSamplerBootstrap(
                n_samples=self.n_bootstrap,
                test_size=self.test_size,
                data=self.data,
                feature_noise_gaussian=-0.1,
            )
        with self.assertRaises(ValueError):
            DataSamplerBootstrap(
                n_samples=self.n_bootstrap,
                test_size=self.test_size,
                data=self.data,
                feature_noise_bernoulli=-0.1,
            )
        with self.assertRaises(ValueError):
            DataSamplerBootstrap(
                n_samples=self.n_bootstrap,
                test_size=self.test_size,
                data=self.data,
                label_shuffle_size=-0.1,
            )

    def test_len(self) -> None:
        """Test __len__ method"""
        self.assertEqual(len(self.data_sampler1), self.n_bootstrap)
        self.assertEqual(len(self.data_sampler2), self.n_bootstrap)

    #     """Test feature size in bootstrap samples"""
    #     train_set, _ = self.data_sampler1[0]
    #     self.assertAlmostEqual(train_set.features.shape[1] / self.n_features, self.feature_size, delta=0.1)

    def test_invalid_parameters(self) -> None:
        """Test invalid parameters for DataSamplerBootstrap"""
        # Test invalid n_samples
        with self.assertRaises(ValueError):
            DataSamplerBootstrap(n_samples=0, test_size=self.test_size, data=self.data)
        with self.assertRaises(ValueError):
            DataSamplerBootstrap(n_samples=-1, test_size=self.test_size, data=self.data)

        # Test invalid test_size
        with self.assertRaises(ValueError):
            DataSamplerBootstrap(
                n_samples=self.n_bootstrap, test_size=-0.1, data=self.data
            )
        with self.assertRaises(ValueError):
            DataSamplerBootstrap(
                n_samples=self.n_bootstrap, test_size=1.1, data=self.data
            )

        # Test invalid feature_size
        with self.assertRaises(ValueError):
            DataSamplerBootstrap(
                n_samples=self.n_bootstrap,
                test_size=self.test_size,
                data=self.data,
                feature_size=-0.1,
            )
        with self.assertRaises(ValueError):
            DataSamplerBootstrap(
                n_samples=self.n_bootstrap,
                test_size=self.test_size,
                data=self.data,
                feature_size=1.1,
            )

        # Test invalid feature_noise_gaussian
        with self.assertRaises(ValueError):
            DataSamplerBootstrap(
                n_samples=self.n_bootstrap,
                test_size=self.test_size,
                data=self.data,
                feature_noise_gaussian=-1,
            )

        # Test invalid random_state
        with self.assertRaises(ValueError):
            DataSamplerBootstrap(
                n_samples=self.n_bootstrap,
                test_size=self.test_size,
                data=self.data,
                random_state=-1,
            )

        # Test invalid data type
        with self.assertRaises(ValueError):
            DataSamplerBootstrap(
                n_samples=self.n_bootstrap,
                test_size=self.test_size,
                data="invalid_data",
            )

    def test_subset_features(self) -> None:
        """Test _subset_features method"""
        # Create a sample DataBase object
        features = pd.DataFrame(np.random.rand(100, 1000))
        labels = pd.Series(np.random.randint(0, 2, 100))
        data = DataBase(features=features, labels=labels)

        # Create a DataSamplerBootstrap object
        sampler = DataSamplerBootstrap(
            n_samples=5, test_size=0.2, data=data, feature_size=0.5
        )

        # Test _subset_features method
        subset_data = sampler._subset_features(data)

        # Check if the number of features is approximately half of the original
        self.assertEqual(
            subset_data.get_features("raw").shape[1]
            / data.get_features("raw").shape[1],
            0.5,
        )

        # Check if the number of samples remains the same
        self.assertEqual(
            subset_data.get_features("raw").shape[0], data.get_features("raw").shape[0]
        )

        # Check if the labels remain unchanged
        pd.testing.assert_series_equal(subset_data.labels, data.labels)

        # Test with feature_size=1.0 (no subsetting)
        sampler.feature_size = 1.0
        full_data = sampler._subset_features(data)
        self.assertEqual(
            full_data.get_features("raw").shape[1], data.get_features("raw").shape[1]
        )

        # Test with feature_size=0.0 (should raise an error)
        sampler.feature_size = 0.0
        with self.assertRaises(AssertionError):
            sampler._subset_features(data)

        # Test with MultiOmicsData
        multi_omics_data = MultiOmicsData(
            gene_expression=features, protein_expression=features, labels=labels
        )
        sampler.feature_size = 0.5
        subset_multi_omics = sampler._subset_features(multi_omics_data)

        # Check if each omics data type has approximately half the features
        for omics_type in subset_multi_omics.multi_omics_data.keys():
            self.assertAlmostEqual(
                subset_multi_omics.multi_omics_data[omics_type]
                .get_features("raw")
                .shape[1]
                / multi_omics_data.multi_omics_data[omics_type]
                .get_features("raw")
                .shape[1],
                0.5,
                delta=0.1,
            )

    def test_add_label_noise(self) -> None:
        """Test _add_label_noise method"""
        # Create a sample DataBase object with regression labels
        features = pd.DataFrame(np.random.rand(100, 10))
        labels = pd.Series(np.random.rand(100))
        data = DataBase(features=features, labels=labels)

        # Create a DataSamplerBootstrap object
        sampler = DataSamplerBootstrap(
            n_samples=5,
            test_size=0.2,
            data=data,
            label_shuffle_size=0.1,
            feature_noise_gaussian=0.05,
        )

        # Test _add_label_noise method
        noisy_data = sampler._add_label_noise(data)

        # Check if the number of samples remains the same
        self.assertEqual(noisy_data.get_labels().shape[0], data.get_labels().shape[0])

        # Check if labels have been modified
        num_different = np.sum(noisy_data.get_labels() != data.get_labels())
        # Assert that some labels have been modified
        self.assertGreater(
            num_different, 0, "No labels were modified by _add_label_noise"
        )

        # Test with classification labels
        labels_class = pd.Series(np.random.randint(0, 2, 100))
        data_class = DataBase(features=features, labels=labels_class)

        noisy_data_class = sampler._add_label_noise(data_class)

        # Check if some labels have been shuffled
        num_different = np.sum(noisy_data_class.get_labels() != data_class.get_labels())
        expected_different = int(
            0.1 * len(labels_class)
        )  # 10% of labels should be different
        self.assertAlmostEqual(
            num_different,
            expected_different,
            delta=self.n_samples * self.label_shuffle_size,
        )  # Allow small deviation

        # Test with label_shuffle_size = 0
        sampler.label_shuffle_size = 0
        no_noise_data = sampler._add_label_noise(data_class)
        pd.testing.assert_series_equal(
            no_noise_data.get_labels(), data_class.get_labels()
        )

        # Test with MultiOmicsData
        multi_omics_data = MultiOmicsData(
            gene_expression=features, protein_expression=features, labels=labels
        )
        noisy_multi_omics = sampler._add_label_noise(multi_omics_data)

        # Check if labels have been modified for MultiOmicsData
        self.assertFalse(
            np.array_equal(
                noisy_multi_omics.get_labels(), multi_omics_data.get_labels()
            )
        )

    def test_add_feature_noise(self) -> None:
        """Test _add_feature_noise method"""
        # Create a sample DataBase object
        features = pd.DataFrame(np.random.rand(100, 10))
        labels = pd.Series(np.random.rand(100))
        data = ContinousData(features=features, labels=labels)

        # Create a DataSamplerBootstrap object
        sampler = DataSamplerBootstrap(
            n_samples=5,
            test_size=0.2,
            data=data,
            feature_noise_gaussian=0.05,
            feature_noise_bernoulli=0.1,
        )

        # Test _add_feature_noise method
        noisy_data = sampler._add_feature_noise(data)

        # Check if the number of samples and features remains the same
        self.assertEqual(
            noisy_data.get_features("raw").shape, data.get_features("raw").shape
        )

        # Check if features have been modified
        self.assertFalse(
            np.array_equal(noisy_data.get_features("raw"), data.get_features("raw"))
        )

        # Check if continuous data (gene_expression) has Gaussian noise
        gene_diff = noisy_data.get_features("raw") - data.get_features("raw")
        self.assertGreater(np.abs(gene_diff).mean().mean(), 0)

        # Test with MultiOmicsData
        # Add mutation data to multi_omics_data
        mutation_data = pd.DataFrame(np.random.randint(0, 2, size=(100, 5)))
        gene_expression = pd.DataFrame(np.random.rand(100, 5))
        protein_expression = pd.DataFrame(np.random.randint(0, 2, size=(100, 5)))
        multi_omics_data = MultiOmicsData(
            gene_expression=gene_expression,
            protein_expression=protein_expression,
            gene_mutation=mutation_data,
            labels=labels,
        )
        noisy_multi_omics = sampler._add_feature_noise(multi_omics_data)

        # Check if features have been modified for MultiOmicsData
        self.assertFalse(
            np.array_equal(
                noisy_multi_omics.get_features(),
                multi_omics_data.get_features(),
            )
        )

        # Check if continuous data (gene_expression) has Gaussian noise
        gene_diff = noisy_multi_omics.get_single_omics_features(
            "gene_expression"
        ) - multi_omics_data.get_single_omics_features("gene_expression")
        self.assertGreater(np.abs(gene_diff).mean().mean(), 0)

        # Check if binary data (gene_mutation) has been flipped
        protein_diff = noisy_multi_omics.get_single_omics_features(
            "gene_mutation"
        ) != multi_omics_data.get_single_omics_features("gene_mutation")
        self.assertGreater(protein_diff.sum().sum(), 0)

        # Test with feature_noise_gaussian = 0 and feature_noise_bernoulli = 0
        sampler.feature_noise_gaussian = 0
        sampler.feature_noise_bernoulli = 0
        no_noise_data = sampler._add_feature_noise(data)
        pd.testing.assert_frame_equal(
            no_noise_data.get_features("raw"), data.get_features("raw")
        )

    def test_resplit(self):
        # Create sample data
        X = pd.DataFrame(np.random.rand(1000, 5))
        y = pd.Series(np.random.uniform(0, 1, 1000))
        data = ContinousData(features=X, labels=y)

        n_samples = 5
        test_size = 0.2
        feature_size = 0.8
        feature_noise_gaussian = 0.05
        feature_noise_bernoulli = 0.1
        label_shuffle_size = 0.1

        # Create a DataSamplerBootstrap object
        sampler = DataSamplerBootstrap(
            n_samples=n_samples,
            test_size=test_size,
            data=data,
            feature_size=feature_size,
            feature_noise_gaussian=feature_noise_gaussian,
            feature_noise_bernoulli=feature_noise_bernoulli,
            label_shuffle_size=label_shuffle_size,
            random_state=42,
        )

        # Store original splits
        original_splits = sampler.splits.copy()

        # Call resplit with a new random state
        sampler.resplit(random_state=100)

        # Check if the number of splits remains the same
        self.assertEqual(len(sampler.splits), n_samples)

        # Check if the splits have changed
        for new_split, original_split in zip(sampler.splits, original_splits):
            self.assertNotEqual(new_split[0].tolist(), original_split[0].tolist())
            self.assertNotEqual(new_split[1].tolist(), original_split[1].tolist())

        # Check if the new splits have the correct structure
        for train_idx, test_idx in sampler.splits:
            self.assertGreaterEqual(len(train_idx) + len(test_idx), len(data))
            self.assertGreaterEqual(len(test_idx), int(test_size * len(data)))

        # Test reproducibility
        sampler1 = DataSamplerBootstrap(
            n_samples=n_samples, test_size=test_size, data=data, random_state=42
        )
        sampler2 = DataSamplerBootstrap(
            n_samples=n_samples, test_size=test_size, data=data, random_state=42
        )

        # Check if each split in sampler1 and sampler2 are equal
        for split1, split2 in zip(sampler1.splits, sampler2.splits):
            np.testing.assert_array_equal(split1[0], split2[0])  # Compare train indices
            np.testing.assert_array_equal(split1[1], split2[1])  # Compare test indices

        sampler1.resplit(random_state=100)
        sampler2.resplit(random_state=100)

        # Check if each split in sampler1 and sampler2 are equal
        for split1, split2 in zip(sampler1.splits, sampler2.splits):
            np.testing.assert_array_equal(split1[0], split2[0])  # Compare train indices
            np.testing.assert_array_equal(split1[1], split2[1])  # Compare test indices

        # Test with different random states
        sampler1.resplit(random_state=101)
        sampler2.resplit(random_state=102)

        # Check if each split in sampler1 and sampler2 are not equal
        for split1, split2 in zip(sampler1.splits, sampler2.splits):
            self.assertFalse(
                np.array_equal(split1[0], split2[0])
            )  # Compare train indices
            self.assertFalse(
                np.array_equal(split1[1], split2[1])
            )  # Compare test indices

    def test_getitem(self):
        # Create sample data
        X = pd.DataFrame(np.random.rand(1000, 5))
        y = pd.Series(np.random.uniform(0, 1, 1000))
        data = ContinousData(features=X, labels=y)

        n_samples = 5
        test_size = 0.2
        feature_size = 0.8
        feature_noise_gaussian = 0.05
        feature_noise_bernoulli = 0.1
        label_shuffle_size = 0.1

        # Create a DataSamplerBootstrap object
        sampler = DataSamplerBootstrap(
            n_samples=n_samples,
            test_size=test_size,
            data=data,
            feature_size=feature_size,
            feature_noise_gaussian=feature_noise_gaussian,
            feature_noise_bernoulli=feature_noise_bernoulli,
            label_shuffle_size=label_shuffle_size,
            random_state=42,
        )

        # Test __getitem__ method
        for i in range(n_samples):
            train_data, test_data = sampler[i]

            # Check if the returned objects are of the correct type
            self.assertIsInstance(train_data, ContinousData)
            self.assertIsInstance(test_data, ContinousData)

            # Check if feature subsetting is applied
            self.assertAlmostEqual(
                train_data.get_shape()[1] / data.get_shape()[1], 0.8, delta=0.1
            )

            # Check if the test data is not modified
            pd.testing.assert_frame_equal(
                test_data.get_features("raw"),
                data.get_features("raw").iloc[sampler.splits[i][1]],
            )

            # Check if noise has been added to training data
            self.assertFalse(
                np.array_equal(
                    train_data.get_features("raw"),
                    data.get_features("raw").iloc[sampler.splits[i][0]],
                )
            )

            # Check if labels have been shuffled in training data
            labels = train_data.get_labels().copy()
            labels.index = data.get_labels().iloc[sampler.splits[i][0]].index
            shuffled_labels = (
                labels != data.get_labels().iloc[sampler.splits[i][0]]
            ).sum()
            expected_shuffled = int(len(train_data) * sampler.label_shuffle_size)
            self.assertGreater(shuffled_labels, 0)

        X = pd.DataFrame(np.random.randint(0, 2, (1000, 5)))
        y = pd.Series(np.random.randint(0, 2, 1000))

        # Test with X and y instead of data
        sampler_xy = DataSamplerBootstrap(
            n_samples=n_samples,
            test_size=test_size,
            X=X,
            y=y,
            feature_size=feature_size,
            feature_noise_gaussian=feature_noise_gaussian,
            feature_noise_bernoulli=feature_noise_bernoulli,
            label_shuffle_size=label_shuffle_size,
            random_state=42,
        )

        for i in range(n_samples):
            X_train, X_test, y_train, y_test = sampler_xy[i]

            # Check if the returned objects are of the correct type
            self.assertIsInstance(X_train, pd.DataFrame)
            self.assertIsInstance(X_test, pd.DataFrame)
            self.assertIsInstance(y_train, pd.Series)
            self.assertIsInstance(y_test, pd.Series)

            # Check if feature subsetting is applied
            self.assertAlmostEqual(X_train.shape[1] / X.shape[1], 0.8, delta=0.1)

            # Check if noise has been added to training data
            self.assertFalse(np.array_equal(X_train, X.iloc[sampler_xy.splits[i][0]]))

            # Check if labels have been shuffled in training data
            shuffled_labels = (y_train != y.iloc[sampler_xy.splits[i][0]]).sum()
            expected_shuffled = int(len(y_train) * sampler_xy.label_shuffle_size)
            self.assertGreater(shuffled_labels, 0)

        # Test with data
        with self.assertRaises(IndexError):
            _ = self.data_sampler1[self.n_bootstrap]

        # Test with X and y
        with self.assertRaises(IndexError):
            _ = self.data_sampler2[self.n_bootstrap]


def main() -> None:
    unittest.main()


if __name__ == "__main__":
    main()