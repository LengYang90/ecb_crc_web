#!/usr/bin/env python3
###############################################################
## Copyright: PromptBio Corp 2024
## Author: Wenhao Gu, Jiayu Chen
## Date of creation: 07/24/2024
## Date of revision: 08/06/2024
#
## Project: MLGenie
## Description: This file defines the model class for MLGenie.
##
###############################################################

#!/usr/bin/env python3

import os
from enum import Enum
from typing import List, Dict, Tuple, Union

import pickle
import random
import scipy
import numpy as np
import pandas as pd
from abc import abstractmethod
from sklearn.base import TransformerMixin
import seaborn as sns
from scipy.spatial import distance
from sklearn.decomposition import PCA as PCAModel
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from .base import ModelBase


class UnsupervisedModel(ModelBase, TransformerMixin):
    """
    Class for EDA Base Model
    """

    model_name = None
    model_type = None

    def __init__(
        self,
        random_state: int = 123,
        path_to_store_model: str = None,
    ):
        super().__init__(
            random_state=random_state,
            path_to_store_model=path_to_store_model,
        )
        self.model = None

    @abstractmethod
    def fit(self, X: pd.DataFrame):
        assert isinstance(X, pd.DataFrame), "Input X should be of type of pd.DataFrame"
        assert not X.empty, "Input X is empty!"

    @abstractmethod
    def transform(self):
        pass

    def _get_hyper_params_space(self):
        pass

    def _HPO(self):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass

    def evaluate(self):
        pass


class PCA(UnsupervisedModel):
    """
    Class for Principal Component Analysis
    """

    model_name = "Principal Component Analysis"
    model_type = "PCA"

    def __init__(
        self,
        dimensionality_reduction: str = "PCA",
        n_components: str = 2,
        random_state: int = 123,
        path_to_store_model: str = None,
    ):
        """
        Initializes the PCA class.

        Params:
            n_components(int): Number of components to keep in PCA algorithm.
            dimensionality_reduction(str): Type of dimensionality_reduction method.
            path_to_store_model (str = None): Path to store the model.
        """
        super().__init__(
            random_state=random_state,
            path_to_store_model=path_to_store_model,
        )
        assert n_components > 1, "n_components must be larger than 1"

        self.dimensionality_reduction = dimensionality_reduction
        self.n_components = n_components
        self.model = PCAModel(n_components=self.n_components, random_state=random_state)
        self.n_features = None

    def fit(self, X: pd.DataFrame) -> None:
        """
        Fit the model according to the given training data.

        Parameters:
            X (pd.DataFrame): The feature matrix.

        Returns:
            self: Returns the instance itself.
        """
        super().fit(X)

        n_samples, n_features = X.shape
        assert self.n_components <= min(
            n_features, n_samples
        ), "n_components must be larger than 1 and smaller than n_features {}".format(
            n_features
        )

        self.model.fit(X)
        self.n_features = n_features
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce X to its most informative features.

        Parameters:
            X (pd.DataFrame): The feature matrix.

        Returns:
            pca_matrix(pd.DataFrame): Projection of the data in the learned components.
            pca_variance_ratio(pd.DataFrame): The proportion of variance explained by each principal component.

        """
        assert (
            self.n_features == X.shape[1]
        ), "X has n_features {}, but PCA is expecting {} features as input".format(
            str(X.shape[1]), str(self.n_features)
        )
        pca_matrix = pd.DataFrame(self.model.transform(X))
        pca_variance_ratio = pd.DataFrame(self.model.explained_variance_ratio_)
        return pca_matrix, pca_variance_ratio


class HC(UnsupervisedModel):
    """
    Class for Hierarchical Clustering
    """

    model_name = "Hierarchical Clustering"
    model_type = "HC"

    def __init__(
        self,
        clusters_limit: Tuple[int, int] = (2, 6),
        n_clusters: int = 2,
        random_state: int = 123,
        path_to_store_model: str = None,
    ):
        """
        Initializes the Hierarchical Clustering class.

        Params:
            clusters_limit(Tuple[int, int]): Upper and lower limits on cluster class.
            n_clusters(int): The number of clusters to find.
            path_to_store_model (str = None): Path to store the model.
        """
        super().__init__(
            random_state=random_state,
            path_to_store_model=path_to_store_model,
        )

        assert n_clusters is None or (
            isinstance(n_clusters, int) and n_clusters > 1
        ), "n_clusters must be larger than 1 or None"

        assert (clusters_limit is None) or (
            isinstance(clusters_limit, Tuple)
            and clusters_limit[1] > clusters_limit[0]
            and clusters_limit[0] >= 2
        ), "clusters_limit {} Error!".format(str(clusters_limit))

        self.clusters_limit = clusters_limit
        self.n_clusters = n_clusters
        self.best_n_clusters = None
        self.model = AgglomerativeClustering(n_clusters=self.n_clusters)

    def fit(self, X: pd.DataFrame) -> None:
        """
        Find the best n_clusters according to the given training data and initialize the model.

        Parameters:
            X (pd.DataFrame): The feature matrix.

        Returns:
            self: Returns the instance itself.
        """
        n_samples, n_features = X.shape
        assert (
            self.n_clusters <= n_samples
        ), "n_clusters {} must be smaller than n_samples {}".format(
            self.n_clusters, n_samples
        )
        if self.clusters_limit:
            best_n_clusters = self._select_best_n_clusters(X)
            if best_n_clusters is None:
                raise ValueError(
                    "clusters_limit is {}, but n_samples is {}, so best_n_clusters is None! Failed to perform Hierarchical Clustering".format(
                        str(self.clusters_limit), str(n_samples)
                    )
                )
            self.best_n_clusters = best_n_clusters
            self.model = AgglomerativeClustering(n_clusters=self.best_n_clusters)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        perform the clustering algorithm

        Parameters:
            X (pd.DataFrame): The feature matrix.

        Returns:
            cluster_labels(pd.DataFrame): Cluster labels for each point.
            cluster_tree_structure(pd.DataFrame): The 2D matrix that holds the children of each non-leaf node.
        """
        fit_result = self.model.fit(X)
        cluster_labels = pd.DataFrame(fit_result.labels_)
        cluster_tree_structure = pd.DataFrame(fit_result.children_)
        return cluster_labels, cluster_tree_structure

    def _select_best_n_clusters(self, X: pd.DataFrame) -> int:
        """
        Select the best n_clusters according to the given data and clusters_limit range.

        Parameters:
            X (pd.DataFrame): The feature matrix.

        Returns:
            int: The best n_clusters.
        """
        n_samples, n_features = X.shape
        max_silhouette_avg = 0.0
        best_n_clusters = None
        for n_clusters in range(self.clusters_limit[0], self.clusters_limit[1]):
            if n_clusters > n_samples - 1:
                continue
            cluster_fit_result = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
            labels = cluster_fit_result.labels_
            silhouette_avg = silhouette_score(X, labels)

            if silhouette_avg > max_silhouette_avg:
                best_n_clusters = n_clusters
                max_silhouette_avg = silhouette_avg
        return best_n_clusters


class FA(UnsupervisedModel):
    """
    Class for Feature Analysis
    """

    model_name = "Feature Analysis"
    model_type = "FA"

    def __init__(
        self,
        top_k: int = 1000,
        random_state: int = 123,
        path_to_store_model: str = None,
    ):
        """
        Initializes the Feature Analysis class.

        Params:
            top_k(int): Keep the k features with the largest variance for feature_matrix in feature analysis.
            path_to_store_model (str = None): Path to store the model.
        """
        super().__init__(
            random_state=random_state,
            path_to_store_model=path_to_store_model,
        )
        assert isinstance(top_k, int) and top_k >= 1, "top_k must be larger than 1"
        self.top_k = top_k
        self.model = None

    def fit(self, X: pd.DataFrame):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature analysis according to the given data.

        There are four steps:
            1. Keep the top-k features with the largest variance
            2. Calculate the pearson correlation matrix as distance matrix
            3. Cluster
            4. Reorder according to cluster map

        Parameters:
            X (pd.DataFrame): The feature matrix.

        Returns:
            feature_corr_coef(pd.DataFrame): the reordered pearson correlation matrix after clustering
        """
        feature_matrix = X.copy(deep=True)

        # 1. Calculate the variance of each  features and retain the largest top_K ones
        top_k_index = (
            feature_matrix.var().sort_values(ascending=False)[: self.top_k].index
        )
        filtered_feature_matrix = feature_matrix[top_k_index]

        # 2. Calculate the pearson correlation matrix
        feature_corr_coef = filtered_feature_matrix.corr(method="pearson").fillna(0)

        # 3. Cluster using feature_corr_coef as distance matrix
        distArray = scipy.spatial.distance.squareform(
            feature_corr_coef.abs(), checks=False
        )
        distLinkage = scipy.cluster.hierarchy.linkage(distArray)
        ax = sns.clustermap(
            feature_corr_coef, row_linkage=distLinkage, col_linkage=distLinkage
        )

        # 4. reorder the index according to clustermap
        origin_index = feature_corr_coef.index
        row_ind = ax.dendrogram_row.reordered_ind
        reordered_index = [origin_index[ind] for ind in row_ind]

        feature_corr_coef = feature_corr_coef.reindex(
            index=reordered_index, columns=reordered_index
        )
        return feature_corr_coef

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass


class EdaModelType(Enum):
    """
    Definition of ModelType of EDA.
    """

    PCA = PCA()
    HC = HC()
    FA = FA()
