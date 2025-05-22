#!/usr/bin/env python3
###############################################################
## Copyright: PromptBio Corp 2022
# Author: jiayuchen
# Date of creation: 07/25/2024
# Date of revision: 12/29/2024
#
## MLGenie
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

from MLGenie.Data import (
    DataBase,
    BinaryData,
    ContinousData,
    OmicsData,
    TimeSeriesData,
    MetaData,
    GeneMutationData,
    GeneExpressionData,
    ProteinExpressionData,
    GeneCNVData,
    GeneMethylationData,
    MultiOmicsData,
)


class TestDataBase(unittest.TestCase):
    """Test DataBase class"""

    def setUp(self) -> None:
        features = {"A": [1, 4, 7, 11], "B": [2, 5, 8, 9], "C": [3, 6, 9, 1]}
        self.labels = pd.Series([1, 1, 0, 0], index=["a", "b", "c", "d"])
        self.features = pd.DataFrame(features, index=["a", "b", "c", "d"])
        self.data = DataBase(features=self.features, labels=self.labels)

    def test_check_data_validity(self) -> None:
        """Test _check_data_validity function"""
        features = [[1, 2, 4], [1, 3, 5], [9, 4, 1]]
        labels = []
        with self.assertRaises(AssertionError):
            data = DataBase(features=features)

        with self.assertRaises(AssertionError):
            data = DataBase(features=pd.DataFrame(features), labels=labels)

        with self.assertRaises(AssertionError):
            data = DataBase(features=pd.DataFrame())

        with self.assertRaises(AssertionError):
            data = DataBase(features=None)

        ############ Test merging different batches of data ###############
        features = []
        labels = []
        n_features = 4
        n_samples = 3
        list_len = 10
        for i in range(list_len):
            index = [str(j) for j in range(i * n_samples, (i + 1) * n_samples)]
            features.append(
                pd.DataFrame(np.random.rand(n_samples, n_features), index=index)
            )
            labels.append(pd.Series([0] * n_samples, index=index))

        # Features and labels are both list
        data = DataBase(features=features, labels=labels)
        self.assertEqual(
            data.get_features("raw").shape, (list_len * n_samples, n_features)
        )
        self.assertEqual(data.get_labels().shape, (list_len * n_samples,))

        # Features is a DataFrame, labels is a list
        data = DataBase(features=features[0], labels=labels[:1])
        self.assertEqual(data.get_features("raw").shape, (n_samples, n_features))
        self.assertEqual(data.get_labels().shape, (n_samples,))

        # Features is a list, labels is a Series
        data = DataBase(features=features[:1], labels=labels[0])
        self.assertEqual(data.get_features("raw").shape, (n_samples, n_features))
        self.assertEqual(data.get_labels().shape, (n_samples,))

        # Features is a list, labels is None
        data = DataBase(features=features[:2], labels=None)
        self.assertEqual(data.get_features("raw").shape, (n_samples * 2, n_features))

        # Bad case 1: features is an empty list
        with self.assertRaises(AssertionError):
            data = DataBase(features=[])

        # Bad case 2: labels is an empty list
        with self.assertRaises(AssertionError):
            data = DataBase(features=features, labels=[])

        # # Bad case 3: Features index not equal to labels index
        # with self.assertRaises(ValueError):
        #     data = DataBase(features=features, labels=labels[:-1])

        # Test with invalid feature value types.
        features = pd.DataFrame(
            {
                "A": ["apple", np.nan, "cherry", "date", "elderberry"],
                "B": [1, 2, 3, 4, 5],
                "C": [1.1, 2.2, np.nan, 4.4, 5.5],
                "D": ["1.1", "2.2", "3.3", "4.4", np.nan],
                "E": [True, True, False, False, True],
            }
        )
        data = DataBase(features=features)
        self.assertListEqual(data.get_features("raw").columns.to_list(), ["B", "C", "E"])


    def test_get_shape(self) -> None:
        """Test get_shape function"""
        shape = self.data.get_shape()
        self.assertEqual(shape, self.features.shape)

    def test_get_features(self) -> None:
        """Test get_features functions"""
        features = self.data.get_features(layer="raw")
        self.assertIsInstance(features, pd.DataFrame)
        self.assertTrue(features.equals(self.features))

        features = self.data.get_features(layer="processed")
        self.assertTrue(features is None)

        with self.assertRaises(ValueError):
            features = self.data.get_features(layer="bad_layer_name")

        selected_features = ["B"]
        features = self.data.get_features(
            layer="raw", selected_features=selected_features
        )
        assert_frame_equal(features, pd.DataFrame(self.features[["B"]]))

        selected_features = ["A", "B"]
        features = self.data.get_features(
            layer="raw", selected_features=selected_features
        )
        assert_frame_equal(features, pd.DataFrame(self.features[["A", "B"]]))

        selected_features = pd.Series(["A"])
        features = self.data.get_features(
            layer="raw", selected_features=selected_features
        )
        assert_frame_equal(features, pd.DataFrame(self.features[["A"]]))

        selected_features = pd.Series(["A", "B"])
        features = self.data.get_features(
            layer="raw", selected_features=selected_features
        )
        assert_frame_equal(features, pd.DataFrame(self.features[["A", "B"]]))

    def test_get_labels(self) -> None:
        labels = self.data.get_labels()
        self.assertIsInstance(labels, pd.Series)
        self.assertTrue(labels.equals(self.labels))

    def test_get_feature_names(self) -> None:
        """Test get_feature_names function"""
        feature_names = self.data.get_feature_names()
        self.assertTrue(self.features.columns.equals(feature_names))

    def test_get_sample_names(self) -> None:
        """Test get_sample_names function"""
        sample_names = self.data.get_sample_names()
        self.assertTrue(self.features.index.equals(sample_names))

    def test_update_features(self) -> None:
        """Test update_features function"""

        data = DataBase(features=self.features, labels=None)

        # Test case 1: update raw layer
        layer = "raw"
        shape = (10, 20)
        features = pd.DataFrame(np.random.rand(shape[0], shape[1]))

        data.update_features(features=features, layer=layer)
        self.assertEqual(shape, data.get_shape())

        feat = pd.DataFrame(
            np.random.rand(int(shape[0] / 2), shape[1]),
            index=[str(i) for i in range(int(shape[0] / 2))],
        )
        feat2 = pd.DataFrame(
            np.random.rand(int(shape[0] / 2), shape[1]),
            index=[str(i) for i in range(int(shape[0] / 2), shape[0])],
        )
        features = [feat, feat2]
        data.update_features(features=features, layer=layer)
        self.assertEqual(shape, data.get_shape())

        # Test case 2: update raw layer
        layer = "processed"
        shape = (20, 10)
        features = pd.DataFrame(np.random.rand(shape[0], shape[1]))
        updated_data = data.update_features(
            features=features, layer=layer, inplace=True
        )
        self.assertEqual(shape, data.get_features("processed").shape)
        self.assertTrue(data is updated_data)

        updated_data = data.update_features(
            features=features, layer=layer, inplace=False
        )
        self.assertEqual(shape, data.get_features("processed").shape)
        self.assertFalse(data is updated_data)

        # # Test case 3: new feature index not equal to labels index
        # with self.assertRaises(ValueError):
        #     self.data.update_features(features=features, layer=layer)

        # Test case 4: empty features
        with self.assertRaises(AssertionError):
            self.data.update_features(features=pd.DataFrame(), layer=layer)

        with self.assertRaises(AssertionError):
            self.data.update_features(features=[], layer=layer)

        # Test case 5: bad layer
        layer = "not_existed"
        with self.assertRaises(ValueError):
            data.update_features(features=features, layer=layer)

    def test_update_labels(self) -> None:
        """Test update_labels function"""

        data = DataBase(features=self.features, labels=None)

        # Test case 1: update raw layer
        layer = "raw"
        data = DataBase(features=self.features)
        updated_data = data.update_labels(labels=self.labels, inplace=True)
        self.assertTrue(data is updated_data)

        updated_data = data.update_labels(labels=self.labels, inplace=False)
        self.assertFalse(data is updated_data)

    def test_update_processed_flag(self) -> None:
        """Test update_processed_flag function"""
        data = DataBase(features=self.features, labels=None)
        data.update_features(features=self.features, layer="processed")
        updated_data = data.update_processed_flag(inplace=True)
        self.assertEqual(data.if_processed, True)
        self.assertTrue(data is updated_data)

        updated_data = data.update_processed_flag(inplace=False)
        self.assertEqual(data.if_processed, True)
        self.assertFalse(data is updated_data)

    def test_drop_duplicate(self) -> None:
        """Test drop_duplicate function"""
        raw_data = {
            "A": [1, 4, 7, 11],
            "B": [2, 5, 8, 9],
            "C": [3, 6, 9, 1],
            "C": [3, 6, 9, 1],
        }
        raw_features = pd.DataFrame(raw_data, index=["a", "a", "b", "c"])
        gold_features = pd.DataFrame(
            {"A": [1, 7, 11], "B": [2, 8, 9], "C": [3, 9, 1]}, index=["a", "b", "c"]
        )
        features = self.data._drop_duplicates(raw_features)
        self.assertTrue(features.equals(gold_features))

    def test_getitem(self) -> None:
        """Test getitem function"""
        index = 1
        data = self.data[index]
        shape = data.get_shape()
        self.assertEqual(shape, (1, self.data.get_shape()[1]))

        data = self.data[:3]
        shape = data.get_shape()
        self.assertEqual(shape, (self.data.get_shape()[1], 3))

        data = self.data[:]
        shape = data.get_shape()
        self.assertEqual(shape, self.data.get_shape())

        data = self.data["a"]
        shape = data.get_shape()
        self.assertEqual(shape, (1, 3))

        data = self.data[["a", "b"]]
        shape = data.get_shape()
        self.assertEqual(shape, (2, 3))

        index = pd.Series(["a", "b", np.nan], index=["a", "b", "d"])
        data = self.data[index]
        shape = data.get_shape()
        self.assertEqual(shape, (3, 3))

        # # Test with 1D numpy array
        # index_1d = np.array(["a", "c"])
        # print(self.features)
        # data_1d = self.data[index_1d]
        # self.assertEqual(data_1d.get_shape(), (2, 3))
        # assert_frame_equal(data_1d.get_features("raw"), self.features.loc[["a", "c"]])

        # # Test with boolean numpy array
        # bool_index = np.array([True, False, True, False])
        # data_bool = self.data[bool_index]
        # self.assertEqual(data_bool.get_shape(), (2, 3))
        # assert_frame_equal(data_bool.get_features("raw"), self.features.iloc[[0, 2]])

        # Test with empty numpy array
        with self.assertRaises(AssertionError):
            empty_index = np.array([])
            data_empty = self.data[empty_index]
            self.assertEqual(data_empty.get_shape(), (0, 3))
            self.assertTrue(data_empty.get_features().empty)

        # Test with out of bounds index
        with self.assertRaises(IndexError):
            self.data[np.array([4, 5])]

    def test_len(self) -> None:
        """
        Test __len__ function
        """
        data = DataBase(features=self.features, labels=self.labels)
        self.assertEqual(len(data), len(self.features))

    def test_copy(self) -> None:
        """
        Test copy function
        """
        copy_data = self.data.copy()
        self.assertFalse(self.data is copy_data)
        assert_frame_equal(
            self.data.get_features(layer="raw"), copy_data.get_features(layer="raw")
        )
        if self.data.if_processed:
            assert_frame_equal(
                self.data.get_features(layer="processed"),
                copy_data.get_features(layer="processed"),
            )
        assert_series_equal(self.data.get_labels(), copy_data.get_labels())
        assert_index_equal(self.data.get_feature_names(), copy_data.get_feature_names())
        assert_index_equal(self.data.get_sample_names(), copy_data.get_sample_names())
        self.assertEqual(self.data.get_label_type(), copy_data.get_label_type())

    def test_check_label_type(self) -> None:
        """
        Test _check_label_type function
        """
        labels = pd.Series([1, 1, 0, 0], index=["a", "b", "c", "d"])
        data = DataBase(features=self.features, labels=labels)
        label_type = data.get_label_type()
        self.assertEqual(label_type, "classification")

        labels = pd.Series([1, 2, 0, 3], index=["a", "b", "c", "d"])
        data = DataBase(features=self.features, labels=labels)
        label_type = data.get_label_type()
        self.assertEqual(label_type, "regression")

        labels = None
        data = DataBase(features=self.features, labels=labels)
        label_type = data.get_label_type()
        self.assertEqual(label_type, None)


class TestOmicsData(unittest.TestCase):
    """Test OmicsData class"""

    def setUp(self) -> None:
        self.shape = (20, 20)
        self.labels = pd.Series(np.random.randint(0, 2, self.shape[0]))
        self.features = pd.DataFrame(np.random.rand(self.shape[0], self.shape[1]))
        self.data = OmicsData(features=self.features, labels=self.labels)

    def test_init(self) -> None:
        """Test init"""
        organism = "error"
        with self.assertRaises(AssertionError):
            data = OmicsData(
                features=self.features, labels=self.labels, organism=organism
            )


class TestBinaryData(unittest.TestCase):
    """Test BinaryData class"""

    def setUp(self) -> None:
        features = {"A": [1, 0, 1, 0], "B": [0, 0, 1, 1], "C": [1, 1, 0, 1]}
        self.labels = pd.Series([1, 1, 0, 1], index=["a", "b", "c", "d"])
        self.features = pd.DataFrame(features, index=["a", "b", "c", "d"])
        self.data = BinaryData(features=self.features, labels=self.labels)

    def test_init(self) -> None:
        """Test init"""
        features = pd.DataFrame([[0.5, 1], [0, 0]])
        with self.assertRaises(ValueError):
            data = BinaryData(features=features)


class TestContinousData(unittest.TestCase):
    """Test ContinousData class"""

    def setUp(self) -> None:
        features = {"A": [1.7, 0.5, 1, 0], "B": [0, 0.5, 1, 1], "C": [1.6, 1, 0, 1]}
        self.labels = pd.Series([1, 1, 0, 0], index=["a", "b", "c", "d"])
        self.features = pd.DataFrame(features, index=["a", "b", "c", "d"])
        self.data = ContinousData(features=self.features, labels=self.labels)

    def test_init(self) -> None:
        """Test init"""
        features = pd.DataFrame([[0, 1], [0, 0], [np.nan, "a"]])
        with self.assertRaises(ValueError):
            data = ContinousData(features=features)


class TestTimeSeriesData(unittest.TestCase):
    """Test TimeSeriesData class"""

    def setUp(self) -> None:
        self.shape = (20, 20)
        self.labels = pd.Series(np.random.randint(0, 2, self.shape[0]))
        self.features = pd.DataFrame(np.random.rand(self.shape[0], self.shape[1]))

    def test_init(self) -> None:
        """Test init"""
        self.data = TimeSeriesData(features=self.features, labels=self.labels)


class TestMetaData(unittest.TestCase):
    """Test MetaData class"""

    def setUp(self) -> None:
        self.shape = (20, 20)
        self.labels = pd.Series(np.random.randint(0, 2, self.shape[0]))
        self.features = pd.DataFrame(np.random.rand(self.shape[0], self.shape[1]))

    def test_init(self) -> None:
        """Test init"""
        self.data = MetaData(features=self.features, labels=self.labels)


class TestGeneMutationData(unittest.TestCase):
    """Test GeneMutationData class"""

    def setUp(self) -> None:
        self.shape = (20, 20)
        self.labels = pd.Series(np.random.randint(0, 2, self.shape[0]))
        self.features = pd.DataFrame(np.random.randint(0, 2, self.shape))

    def test_init(self) -> None:
        """Test init"""
        self.data = GeneMutationData(features=self.features, labels=self.labels)


class TestGeneExpressionData(unittest.TestCase):
    """Test GeneExpressionData class"""

    def setUp(self) -> None:
        self.shape = (20, 20)
        self.labels = pd.Series(np.random.randint(0, 20, self.shape[0]))
        self.features = pd.DataFrame(np.random.rand(self.shape[0], self.shape[1]))

    def test_init(self) -> None:
        """Test init"""
        self.data = GeneExpressionData(features=self.features, labels=self.labels)

        features = pd.DataFrame([[0, 1, 3, 7]])
        data = GeneExpressionData(features=features, count=True)
        features = data.get_features("raw").values.tolist()
        self.assertEqual(features, [[0.0, 1.0, 2.0, 3.0]])

        features = pd.DataFrame([[-1.5, 1, 3, 7]])
        data = GeneExpressionData(features=features, count=True)
        features = data.get_features("raw").values.tolist()
        self.assertEqual(features, [[-1.5, 1, 3, 7]])


class TestProteinExpressionData(unittest.TestCase):
    """Test ProteinExpressionData class"""

    def setUp(self) -> None:
        self.shape = (20, 20)
        self.labels = pd.Series(np.random.randint(0, 2, self.shape[0]))
        self.features = pd.DataFrame(np.random.rand(self.shape[0], self.shape[1]))

    def test_init(self) -> None:
        """Test init"""
        self.data = ProteinExpressionData(features=self.features, labels=self.labels)


class TestGeneCNVData(unittest.TestCase):
    """Test GeneCNVData class"""

    def setUp(self) -> None:
        self.shape = (20, 20)
        self.labels = pd.Series(np.random.randint(0, 2, self.shape[0]))
        self.features = pd.DataFrame(np.random.randint(0, 100, self.shape), dtype=float)

    def test_init(self) -> None:
        """Test init"""
        self.data = GeneCNVData(features=self.features, labels=self.labels)

        features = pd.DataFrame([[0, 1, 3, 10]])
        data = GeneCNVData(features=features)
        features = data.get_features("raw").values.tolist()
        self.assertEqual(features, [[-2.0, -1, 1, 5]])


class TestGeneMethylationData(unittest.TestCase):
    """Test MetaData class"""

    def setUp(self) -> None:
        self.shape = (20, 20)
        self.labels = pd.Series(np.random.randint(0, 1, self.shape[0]))
        self.features = pd.DataFrame(np.random.uniform(0, 1, self.shape))
        self.data = GeneMethylationData(features=self.features, labels=self.labels)

    def test_init(self) -> None:
        """Test init"""
        features = pd.DataFrame(np.random.uniform(1, 2, self.shape), dtype=float)
        with self.assertRaises(ValueError):
            data = GeneMethylationData(features=features, labels=self.labels)


class TestMultiOmicsData(unittest.TestCase):
    """Test MultiOmicsData class"""

    def setUp(self) -> None:
        self.shape = (20, 20)
        self.index = pd.Index(
            ["s00" + str(i) for i in range(self.shape[0])], name="sample_id"
        )
        self.labels = pd.Series(
            np.random.randint(0, 2, self.shape[0]), index=self.index
        )
        self.gene_mutation = pd.DataFrame(
            np.random.randint(0, 2, self.shape), index=self.index
        )
        self.gene_expression = pd.DataFrame(
            np.random.uniform(-5, 100, self.shape), index=self.index
        )
        self.gene_fusion = pd.DataFrame(
            np.random.uniform(-5, 100, self.shape), index=self.index
        )
        self.gene_cnv = pd.DataFrame(
            np.random.uniform(-5, 100, self.shape), index=self.index
        )
        self.gene_methylation = pd.DataFrame(
            np.random.uniform(0, 1, self.shape), index=self.index
        )
        self.mirna_expression = pd.DataFrame(
            np.random.uniform(-5, 100, self.shape), index=self.index
        )
        self.circrna_expression = pd.DataFrame(
            np.random.uniform(-5, 100, self.shape), index=self.index
        )
        self.protein_expression = pd.DataFrame(
            np.random.uniform(-5, 100, self.shape), index=self.index
        )
        self.protein_phosphorylation = pd.DataFrame(
            np.random.uniform(-5, 100, self.shape), index=self.index
        )
        self.protein_acetylation = pd.DataFrame(
            np.random.uniform(-5, 100, self.shape), index=self.index
        )
        self.time_series_data = pd.DataFrame(
            np.random.uniform(-5, 100, self.shape), index=self.index
        )
        self.meta_data = pd.DataFrame(
            np.random.uniform(-5, 100, self.shape), index=self.index
        )
        self.scrna_data = pd.DataFrame(
            np.random.uniform(0, 1, self.shape), index=self.index
        )
        self.metabolomics_data = pd.DataFrame(
            np.random.uniform(0, 1, self.shape), index=self.index
        )
        self.other_data = pd.DataFrame(
            np.random.uniform(-5, 100, self.shape), index=self.index
        )

        self.organism = "human"

        # Test case 1
        if_processed = False
        self.data = MultiOmicsData(
            gene_mutation=self.gene_mutation,
            gene_expression=self.gene_expression,
            gene_fusion=self.gene_fusion,
            gene_cnv=self.gene_cnv,
            gene_methylation=self.gene_methylation,
            mirna_expression=self.mirna_expression,
            circrna_expression=self.circrna_expression,
            protein_expression=self.protein_expression,
            protein_phosphorylation=self.protein_phosphorylation,
            protein_acetylation=self.protein_acetylation,
            time_series_data=self.time_series_data,
            meta_data=self.meta_data,
            scrna_data=self.scrna_data,
            metabolomics_data=self.metabolomics_data,
            other_data=self.other_data,
            labels=self.labels,
            organism=self.organism,
            if_processed=if_processed,
        )

        # Test case 2
        if_processed = True
        concat_mode = "outer"
        self.data_2 = MultiOmicsData(
            gene_mutation=self.gene_mutation,
            gene_expression=self.gene_expression,
            gene_fusion=self.gene_fusion,
            gene_cnv=self.gene_cnv,
            gene_methylation=self.gene_methylation,
            mirna_expression=self.mirna_expression,
            circrna_expression=self.circrna_expression,
            protein_expression=self.protein_expression,
            protein_phosphorylation=self.protein_phosphorylation,
            protein_acetylation=self.protein_acetylation,
            time_series_data=self.time_series_data,
            meta_data=self.meta_data,
            scrna_data=self.scrna_data,
            metabolomics_data=self.metabolomics_data,
            other_data=self.other_data,
            labels=self.labels,
            organism=self.organism,
            if_processed=if_processed,
            concat_mode=concat_mode,
        )

        # Test case 3
        if_processed = True
        concat_mode = "outer"
        self.data_2 = MultiOmicsData(
            gene_mutation=[self.gene_mutation],
            gene_expression=self.gene_expression,
            gene_cnv=self.gene_cnv,
            labels=[self.labels],
            organism=self.organism,
            if_processed=if_processed,
            concat_mode=concat_mode,
        )

    def test_concat_samples(self) -> None:
        """
        Test _concat_samples and get_sample_names function
        """
        features = [[0, 1], [1, 1], [0, 0], [1, 0]]
        gene_mutation = pd.DataFrame(features, index=["a01", "a02", "a03", "a04"])
        gene_expression = pd.DataFrame(features, index=["a01", "a03", "a04", "a05"])
        gene_cnv = pd.DataFrame(features, index=["a01", "a03", "a04", "a06"])
        data = MultiOmicsData(
            gene_mutation=gene_mutation,
            gene_expression=gene_expression,
            gene_cnv=gene_cnv,
            concat_mode="inner",
        )
        sample_names = data.get_sample_names()
        self.assertEqual(sample_names.to_list(), ["a01", "a03", "a04"])

        data = MultiOmicsData(
            gene_mutation=gene_mutation,
            gene_expression=gene_expression,
            gene_cnv=gene_cnv,
            concat_mode="outer",
        )
        sample_names = data.get_sample_names()
        self.assertEqual(
            sample_names.to_list(), ["a01", "a02", "a03", "a04", "a05", "a06"]
        )

        # bad case: The intersection of sample names is empty
        gene_mutation = pd.DataFrame(features, index=["a01", "a02", "a03", "a04"])
        gene_expression = pd.DataFrame(features, index=["a05", "a06", "a07", "a08"])
        with self.assertRaises(ValueError):
            data = MultiOmicsData(
                gene_mutation=gene_mutation,
                gene_expression=gene_expression,
                gene_cnv=gene_cnv,
                concat_mode="inner",
            )

        # bad case: error concat mode
        with self.assertRaises(ValueError):
            data = MultiOmicsData(
                gene_mutation=gene_mutation,
                gene_expression=gene_expression,
                gene_cnv=gene_cnv,
                concat_mode="inner_bad",
            )

    def test_concat_features(self) -> None:
        """
        Test _concat_features function
        """
        feature_names = self.data.get_feature_names()
        self.assertEqual(len(feature_names), 15 * 20)

        features = [[0, 1], [1, 1], [0, 0], [1, 0]]
        gene_mutation = pd.DataFrame(features, index=["a01", "a02", "a03", "a04"])
        gene_expression = pd.DataFrame(features, index=["a01", "a03", "a04", "a05"])
        gene_cnv = pd.DataFrame(features, index=["a01", "a03", "a04", "a06"])
        data = MultiOmicsData(
            gene_mutation=gene_mutation,
            gene_expression=gene_expression,
            gene_cnv=gene_cnv,
            concat_mode="inner",
        )
        feature_names = data.get_feature_names()
        self.assertEqual(len(feature_names), 2 * 3)

    def test_get_shape(self) -> None:
        """
        Test get_shape function
        """
        self.assertEqual(self.data.get_shape(), (self.shape[0], self.shape[1] * 15))

    def test_get_labels(self) -> None:
        """
        Test get_labels function
        """
        labels = self.data.get_labels()
        self.assertEqual(len(labels), len(self.labels))

        features = [[0, 1], [1, 1], [0, 0], [1, 0]]
        labels = pd.Series(
            [0, 1, 0, 1, 0, 1, 0],
            index=["a01", "a02", "a03", "a04", "a05", "a06", "a07"],
        )
        gene_mutation = pd.DataFrame(
            features, index=["a01", "a02", "a03", "a04"], columns=["a", "b"]
        )
        gene_expression = pd.DataFrame(features, index=["a01", "a03", "a04", "a05"])
        gene_cnv = pd.DataFrame(features, index=["a01", "a03", "a06", "a07"])
        data = MultiOmicsData(
            gene_mutation=gene_mutation,
            gene_expression=gene_expression,
            gene_cnv=gene_cnv,
            labels=labels,
            concat_mode="inner",
        )
        labels = data.get_labels()
        self.assertEqual(len(labels), len(["a01", "a03"]))

    def test_update_features(self) -> None:
        """
        Test update_features functions
        """
        features = [[0, 1], [1, 1], [0, 0], [1, 0]]
        labels = pd.Series(
            [0, 1, 0, 1, 0, 1], index=["a01", "a02", "a03", "a04", "a05", "a06"]
        )
        gene_mutation = pd.DataFrame(features, index=["a01", "a02", "a03", "a04"])
        gene_expression = pd.DataFrame(features, index=["a01", "a03", "a04", "a05"])
        gene_cnv = pd.DataFrame(features, index=["a01", "a03", "a04", "a06"])
        data = MultiOmicsData(
            gene_mutation=gene_mutation,
            gene_expression=gene_expression,
            gene_cnv=gene_cnv,
            labels=labels,
            concat_mode="inner",
        )

        new_features = [[0, 0], [1, 1], [0, 0], [1, 1]]
        new_gene_mutation = pd.DataFrame(
            new_features, index=["a01", "a02", "a03", "a04"]
        )

        updated_data = data.update_features(
            gene_mutation=new_gene_mutation,
            gene_expression=gene_expression,
            gene_cnv=gene_cnv,
        )
        self.assertTrue(data is updated_data)
        updated_features = data.get_single_omics_features("gene_mutation")
        assert_frame_equal(new_gene_mutation, updated_features)

        updated_data = data.update_features(
            gene_mutation=[new_gene_mutation],
            gene_expression=[gene_expression],
            gene_cnv=[gene_cnv],
        )
        self.assertTrue(data is updated_data)
        updated_features = data.get_single_omics_features("gene_mutation")
        assert_frame_equal(new_gene_mutation, updated_features)

        updated_data = data.update_features(
            gene_mutation=new_gene_mutation,
            gene_expression=gene_expression,
            gene_cnv=gene_cnv,
            inplace=False,
        )
        self.assertFalse(data is updated_data)
        updated_features = data.get_single_omics_features("gene_mutation")
        assert_frame_equal(new_gene_mutation, updated_features)

    def test_update_labels(self) -> None:
        """
        Test update_labels function
        """
        features = [[0, 1], [1, 1], [0, 0], [1, 0]]
        labels = pd.Series([0, 1, 0, 1], index=["a01", "a02", "a03", "a04"])
        gene_mutation = pd.DataFrame(features, index=["a01", "a02", "a03", "a04"])
        data = MultiOmicsData(
            gene_mutation=gene_mutation, labels=labels, concat_mode="inner"
        )
        new_labels = pd.Series([1, 1, 1, 1], index=["a01", "a02", "a03", "a04"])
        updated_data = data.update_labels(labels=new_labels, inplace=True)
        self.assertTrue(data is updated_data)
        assert_series_equal(updated_data.get_labels(), new_labels)

        updated_data = data.update_labels(labels=new_labels, inplace=False)
        self.assertFalse(data is updated_data)
        assert_series_equal(updated_data.get_labels(), new_labels)

    def test_get_single_omics_features(self) -> None:
        """
        Test get_single_omics_features function
        """
        omics_data_type = "gene_mutation"
        features = self.data.get_single_omics_features(omics_data_type)
        assert_frame_equal(features, self.gene_mutation)

        with self.assertRaises(ValueError):
            self.data.get_single_omics_features("bad_key")

        features = [[0, 1], [1, 1], [0, 0], [1, 0]]
        labels = pd.Series([0, 1, 0, 1], index=["a01", "a02", "a03", "a04"])
        gene_mutation = pd.DataFrame(features, index=["a01", "a02", "a03", "a04"])
        data = MultiOmicsData(
            gene_mutation=gene_mutation, labels=labels, concat_mode="inner"
        )
        features = data.get_single_omics_features("gene_cnv")
        self.assertTrue(features is None)

    def test_update_processed_flag(self) -> None:
        """
        Test update_processed_flag
        """
        features = [[0, 1], [1, 1], [0, 0], [1, 0]]
        labels = pd.Series([0, 1, 0, 1], index=["a01", "a02", "a03", "a04"])
        gene_mutation = pd.DataFrame(features, index=["a01", "a02", "a03", "a04"])
        data = MultiOmicsData(
            gene_mutation=gene_mutation, labels=labels, concat_mode="inner"
        )
        for data_type, omics_data in data.multi_omics_data.items():
            omics_data.update_features(gene_mutation)
            omics_data.update_processed_flag()
        updated_data = data.update_processed_flag(inplace=True)
        self.assertTrue(data is updated_data)
        self.assertEqual(data.if_processed, True)

        updated_data = data.update_processed_flag(inplace=False)
        self.assertFalse(data is updated_data)
        self.assertEqual(data.if_processed, True)

    def test_check_data_validity(self) -> None:
        """
        Test _check_data_validity function
        """
        with self.assertRaises(ValueError):
            data = MultiOmicsData()

        # with self.assertRaises(AssertionError):
        #     data = MultiOmicsData(
        #         gene_mutation=[self.gene_mutation],
        #         gene_expression=[],
        #     )

        with self.assertRaises(AssertionError):
            data = MultiOmicsData(
                gene_mutation=[self.gene_mutation], gene_expression=[], labels=[]
            )

        with self.assertRaises(AssertionError):
            data = MultiOmicsData(
                gene_mutation=[self.gene_mutation], gene_expression=[], labels=[0, 1]
            )

        with self.assertRaises(AssertionError):
            data = MultiOmicsData(
                gene_mutation=[0, 0, 0, 1],
            )

    def test_len(self) -> None:
        """
        Test __len__ function
        """
        self.assertEqual(len(self.data), self.shape[0])

    def test_copy(self) -> None:
        """
        Test copy function
        """
        copy_data = self.data.copy()
        self.assertFalse(self.data is copy_data)
        self.assertTrue(isinstance(copy_data, MultiOmicsData))

    def test_check_label_type(self) -> None:
        """
        Test _check_label_type function
        """
        data = MultiOmicsData(gene_mutation=self.gene_mutation, labels=self.labels)
        label_type = data.get_label_type()
        self.assertEqual(label_type, "classification")

        labels = pd.Series(np.random.randint(0, 10, self.shape[0]), index=self.index)
        data = MultiOmicsData(gene_mutation=self.gene_mutation, labels=labels)
        label_type = data.get_label_type()
        self.assertEqual(label_type, "regression")

        labels = None
        data = MultiOmicsData(gene_mutation=self.gene_mutation, labels=labels)
        label_type = data.get_label_type()
        self.assertEqual(label_type, None)

    def test_construct_mapping(self) -> None:
        features = [[0, 1], [1, 1], [0, 0], [1, 0]]
        labels = pd.Series(
            [0, 1, 0, 1, 0, 1], index=["a01", "a02", "a03", "a04", "a05", "a06"]
        )
        gene_mutation = pd.DataFrame(features, index=["a01", "a02", "a03", "a04"])
        gene_expression = pd.DataFrame(features, index=["a01", "a03", "a04", "a05"])
        gene_cnv = pd.DataFrame(features, index=["a01", "a03", "a04", "a06"])
        data = MultiOmicsData(
            gene_mutation=gene_mutation,
            gene_expression=gene_expression,
            gene_cnv=gene_cnv,
            labels=labels,
            concat_mode="inner",
        )

    def test_getitem(self) -> None:
        """
        Test __getitem__ function
        """
        features = [[0, 1], [1, 1], [0, 0], [1, 0]]
        labels = pd.Series(
            [0, 1, 0, 1, 0, 1, 0],
            index=["a01", "a02", "a03", "a04", "a05", "a06", "a07"],
        )
        gene_mutation = pd.DataFrame(
            features, index=["a01", "a02", "a03", "a04"], columns=["a", "b"]
        )
        gene_expression = pd.DataFrame(features, index=["a01", "a03", "a04", "a05"])
        gene_cnv = pd.DataFrame(features, index=["a01", "a03", "a04", "a07"])
        data = MultiOmicsData(
            gene_mutation=gene_mutation,
            gene_expression=gene_expression,
            gene_cnv=gene_cnv,
            labels=labels,
            concat_mode="outer",
        )
        new_data = self.data[1:6]
        self.assertEqual(new_data.get_shape(), (5, 15 * self.shape[1]))

        new_data = self.data[6]
        self.assertEqual(new_data.get_shape(), (1, 15 * self.shape[1]))

    def test_get_features(self) -> None:
        """
        Test get_features function
        """
        combined_features = self.data.get_features()
        self.assertEqual(combined_features.shape, self.data.get_shape())

        features = [[0, 1], [1, 1], [0, 0], [1, 0]]
        labels = pd.Series(
            [0, 1, 0, 1, 0, 1, 0],
            index=["a01", "a02", "a03", "a04", "a05", "a06", "a07"],
        )
        gene_mutation = pd.DataFrame(
            features, index=["a01", "a02", "a03", "a04"], columns=["a", "b"]
        )
        gene_expression = pd.DataFrame(features, index=["a01", "a03", "a04", "a05"])
        gene_cnv = pd.DataFrame(features, index=["a01", "a03", "a06", "a07"])
        data = MultiOmicsData(
            gene_mutation=gene_mutation,
            gene_expression=gene_expression,
            gene_cnv=gene_cnv,
            labels=labels,
            concat_mode="outer",
        )
        combined_features = data.get_features()

        self.assertEqual(combined_features.shape, (len(labels), 3 * 2))

        selected_features = pd.Series(["gene_cnv|0", "gene_expression|1"])
        combined_features = data.get_features(
            layer="raw", selected_features=selected_features
        )


def main() -> None:
    unittest.main()


if __name__ == "__main__":
    main()