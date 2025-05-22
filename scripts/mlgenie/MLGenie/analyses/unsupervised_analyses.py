#!/usr/bin/env python3
###############################################################
## Copyright: PromptBio Corp 2024
## Author: Wenhao Gu, Jiayu Chen
## Date of creation: 07/23/2024
## Date of revision: 12/06/2024
## Project: MLGenie
## Description: This file defines the class for EDA analysis.
##
###############################################################
import os
import sys
from typing import List, Union, Tuple, Dict
from abc import abstractmethod

import numpy as np
import pandas as pd

from ._base import AnalysisBase
from ..Data import MultiOmicsData,DataBase
from ..model.eda import PCA, HC, FA
from ..utils import get_statistic_matrix

class UnsupervisedAnalysis(AnalysisBase):
    def __init__(
        self,
        null_ratio_threshold: float=0.5,
        dominance_threshold: float=0.95,
        impute_strategy: str="constant",
        impute_fill_value: float=0,
        chi2_p_threshold: float=0.5,
        variance_threshold: float=0.01,
        fold_change_threshold: float=None,
        target_corr_threshold: float=0,
        skew_threshold: float=2.5,
        skew_correction_method: str="yeo-johnson",
        outlier_detection_method: str="z_score",
        outlier_detection_threshold: float=3,
        outlier_correction_method: str="clip",
        scaling_method: str="standard",
        scaling_feature_range: Tuple[float, float]=(0, 1),
        feature_corr_threshold: float=0.95,
        n_components: int = 2,
        dimensionality_reduction: str = "PCA",
        top_k: int = 1000,
        n_clusters: int = 2,
        result_dir: str = None,
        random_state: int = 123
        ):
        """
        Initialize the unsupervised analysis base.

        Params:
            n_components: The number of components to keep.
            dimensionality_reduction: The dimensionality reduction method.
            top_k: The number of top features to keep.
            n_clusters: The number of clusters to keep.
            result_dir: The directory to save the result.
            random_state: The random state for the analysis.
        """
        # check n_components
        if not isinstance(n_components, int) or n_components <= 0:
            raise ValueError("n_components must be a positive integer.")

        # check top_k
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        # check dimensionality_reduction
        if not isinstance(dimensionality_reduction, str):
            raise ValueError("dimensionality_reduction must be a string.")

        self.n_components = n_components
        self.dimensionality_reduction = dimensionality_reduction
        self.top_k = top_k
        self.n_clusters = n_clusters
        self.pca_model = None 
        self.cluster_model = None 
        self.feature_analysis_model = None
        super().__init__(
            result_dir=result_dir, 
            random_state=random_state,
            null_ratio_threshold = null_ratio_threshold,
            dominance_threshold = dominance_threshold,
            impute_strategy = impute_strategy,
            impute_fill_value = impute_fill_value,
            chi2_p_threshold = chi2_p_threshold,
            variance_threshold = variance_threshold,
            fold_change_threshold = fold_change_threshold,
            target_corr_threshold = target_corr_threshold,
            skew_threshold = skew_threshold,
            skew_correction_method = skew_correction_method,
            outlier_detection_method = outlier_detection_method,
            outlier_detection_threshold = outlier_detection_threshold,
            outlier_correction_method = outlier_correction_method,
            scaling_method = scaling_method,
            scaling_feature_range = scaling_feature_range,
            feature_corr_threshold = feature_corr_threshold
            )
        self.statistic_matrix = None


    def _check_n_components(self, data: DataBase)-> None:
        n_samples, n_features = data.get_shape()
        self.n_components = min(n_samples, n_features, self.n_components)

    def fit(self, data: DataBase):
        # Calculate statistic matrix
        raw_features = data.get_features("raw")
        self.statistic_matrix = get_statistic_matrix(raw_features)
        
        # Preprocess the data.
        data = self._preprocess(data)
        print("--------_preprocess done!----------------\n\n")

        self._check_n_components(data)
        features = data.get_features("processed")

        # Run PCA
        self.pca_model = PCA(
            dimensionality_reduction = self.dimensionality_reduction,
            n_components = self.n_components,
            random_state = self.random_state
            )
        self.pca_model.fit(features)

        # Run Hierarchical Clustering
        self.cluster_model = HC(
            n_clusters = self.n_clusters,
            random_state = self.random_state
        )
        self.cluster_model.fit(features)

        # Run Feature Analysis
        self.feature_analysis_model = FA(
            top_k = self.top_k,
            random_state = self.random_state
        )
        self.feature_analysis_model.fit(features)

    def transform(self, data: DataBase):
        if not hasattr(self, 'pca_model'):
            raise ValueError("Model has not been fit yet. Please call fit() before transform().")
        # Calculate statistic matrix
        raw_features = data.get_features("raw")
        self.statistic_matrix = get_statistic_matrix(raw_features)
        
        # Preprocess the data.
        data = self._preprocess(data)
        print("--------_preprocess done!----------------\n\n")

        self._check_n_components(data)
        features = data.get_features("processed")
        
        # Run PCA
        pca_matrix, pca_variance_ratio = self.pca_model.transform(features)

        # Run Hierarchical Clustering
        cluster_labels, cluster_tree_structure = self.cluster_model.transform(features)

        # Run Feature Analysis
        feature_corr_coef = self.feature_analysis_model.transform(features)

        result = {
            "pca_matrix": pca_matrix,
            "pca_variance_ratio": pca_variance_ratio,
            "cluster_labels": cluster_labels,
            "cluster_tree_structure": cluster_tree_structure,
            "feature_corr_coef":feature_corr_coef,
            "feature_statistics": self.statistic_matrix,
            "processed_feature_matrix": features
        }
        return result

    def save_result(self, result: Dict):
        """
        Save the result to the result directory.
        """
        # Save processed feature matrix
        processed_feature_matrix = result["processed_feature_matrix"]
        processed_feature_matrix.to_csv(os.path.join(self.result_dir, "processed_feature_matrix.csv"))

        # Save feature statistics
        feature_statistics = result["feature_statistics"]
        feature_statistics.to_csv(os.path.join(self.result_dir, "feature_statistics.csv"), index=False)

        # Save PCA result
        pca_matrix = result["pca_matrix"]
        pca_variance_ratio = result["pca_variance_ratio"]
        pca_matrix.index = processed_feature_matrix.index
        pca_matrix.to_csv(os.path.join(self.result_dir, "pca_matrix.csv"))
        pca_variance_ratio.columns = ["variance_ratio"]
        pca_variance_ratio.to_csv(os.path.join(self.result_dir, "pca_variance_ratio.csv"))

        # Save cluster result
        cluster_labels = result["cluster_labels"]
        cluster_tree_structure = result["cluster_tree_structure"]
        index_map = {
            id: index_name
            for id, index_name in zip(cluster_labels.index, processed_feature_matrix.index)
        }
        cluster_labels.index = processed_feature_matrix.index
        cluster_tree_structure = cluster_tree_structure.replace(index_map)

        cluster_labels.to_csv(os.path.join(self.result_dir, "cluster_labels.csv"))
        cluster_tree_structure.to_csv(os.path.join(self.result_dir, "cluster_tree_structure.csv"))

        # Save feature correlation coefficient
        feature_corr_coef = result["feature_corr_coef"]
        feature_corr_coef.to_csv(os.path.join(self.result_dir, "feature_corr_coef.csv"))
