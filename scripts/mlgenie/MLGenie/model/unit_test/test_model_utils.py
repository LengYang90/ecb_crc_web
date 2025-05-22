#!/usr/bin/env python3
###############################################################
## Copyright: PromptBio Corp 2023
# Author: jiayu
# Date of creation: 09/13/2024
# Date of creation:
#
## AIM
## Description: Unit test for model utility functions
#
###############################################################
import os
import sys
import unittest
from typing import List, Dict, Tuple, Union

import numpy as np
import pandas as pd
from sklearn import datasets

cur_folder = os.path.dirname(os.path.realpath(__file__))
last_folder = os.path.dirname(cur_folder)
last_last_folder = os.path.dirname(last_folder)
last_last_last_folder = os.path.dirname(last_last_folder)
last_last_last_last_folder = os.path.dirname(last_last_last_folder)
sys.path.append(last_last_last_last_folder)

from MLGenie.design.model.model_utils import compare_models
from MLGenie.design.utils import AnalysisType, Metrics
from MLGenie.design.model.classification import SVM, LR, RF, KNN, DT, GBDT, MLP
from MLGenie.design.model.regression import LassoR, RidgeR, ElasticR, GBDTR, RFR, DTR
from MLGenie.design.model.survival import CoxPH, Coxnet, RSF, GBS, FSVM


class TestUtils(unittest.TestCase):
    """Test utility functions"""

    def setUp(self) -> None:
        # Load the iris dataset.
        iris = datasets.load_iris()
        self.X = pd.DataFrame(iris.data)
        self.X.columns = ["f0", "f1", "f2", "f3"]
        self.y = iris.target
    def test_compare_models(self) -> None:
        """
        Test compare_models().
        """
        # Test input X
        result = compare_models(
            self.X,
            self.y,
            analysis_type=AnalysisType.Classification,
            metrics=Metrics.Accuracy,
        )
        self.assertIsInstance(result, List)
        for item in result:
            self.assertIsInstance(item[1], float)
            self.assertTrue(item[1] >= 0)

        with self.assertRaises(TypeError):
            compare_models(
                self.X.values,
                self.y,
                analysis_type=AnalysisType.Classification,
            )

        with self.assertRaises(TypeError):
            compare_models(
                self.X.values.tolist(),
                self.y,
                analysis_type=AnalysisType.Classification,
            )

        with self.assertRaises(TypeError):
            compare_models(
                None,
                self.y,
                analysis_type=AnalysisType.Classification,
            )

        # Test input y
        with self.assertRaises(TypeError):
            compare_models(
                self.X,
                [1, 2, 3],
                analysis_type=AnalysisType.Classification,
            )
        with self.assertRaises(TypeError):
            compare_models(
                self.X,
                None,
                analysis_type=AnalysisType.Classification,
            )

        # Test analysis_type
        compare_models(
            self.X,
            self.y,
            analysis_type=AnalysisType.Regression,
            metrics=Metrics.R2,
        )

        compare_models(
            self.X,
            self.y,
            analysis_type=AnalysisType.Survival,
            metrics=None,
        )
        with self.assertRaises(TypeError):
            compare_models(
                self.X,
                self.y,
                analysis_type="Bad_type",
            )

        # Test specified models
        compare_models(
            self.X,
            self.y,
            analysis_type=AnalysisType.Classification,
            specified_models=["SVM", "LR", "RF", "KNN", "DT", "GBDT", "MLP"],
            metrics=Metrics.Accuracy,
        )

        compare_models(
            self.X,
            self.y,
            analysis_type=AnalysisType.Regression,
            specified_models=["LassoR", "RidgeR", "ElasticR", "GBDTR", "RFR", "DTR"],
            metrics=Metrics.R2,
        )

        compare_models(
            self.X,
            self.y,
            analysis_type=AnalysisType.Survival,
            specified_models=["CoxPH", "Coxnet", "RSF", "GBS", "FSVM"],
        )

        with self.assertRaises(ValueError):
            compare_models(
                self.X,
                self.y,
                analysis_type=AnalysisType.Classification,
                specified_models=["SVM222"],
            )

        with self.assertRaises(ValueError):
            compare_models(
                self.X,
                self.y,
                analysis_type=AnalysisType.Regression,
                specified_models=["SVM222"],
            )

        with self.assertRaises(ValueError):
            compare_models(
                self.X,
                self.y,
                analysis_type=AnalysisType.Survival,
                specified_models=["SVM222"],
            )

        # Test metrics
        with self.assertRaises(TypeError):
            compare_models(
                self.X,
                self.y,
                analysis_type=AnalysisType.Survival,
                metrics="Accuracy",
            )

        # Test average
        with self.assertRaises(TypeError):
            compare_models(
                self.X,
                self.y,
                analysis_type=AnalysisType.Classification,
                average="average",
            )

        # Test cv
        with self.assertRaises(TypeError):
            compare_models(
                self.X, self.y, analysis_type=AnalysisType.Classification, cv=0
            )

        with self.assertRaises(TypeError):
            compare_models(
                self.X, self.y, analysis_type=AnalysisType.Classification, cv="1"
            )

        # Test n_jobs
        with self.assertRaises(TypeError):
            compare_models(
                self.X, self.y, analysis_type=AnalysisType.Classification, n_jobs=-100
            )

        with self.assertRaises(TypeError):
            compare_models(
                self.X, self.y, analysis_type=AnalysisType.Classification, n_jobs="8"
            )


def main() -> None:
    unittest.main()


if __name__ == "__main__":
    main()
