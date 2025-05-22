#!/usr/bin/env python3
###############################################################
## Copyright: PromptBio Corp 2024
## Author: Minzhe Zhang, Wenhao Gu
## Date of creation: 07/18/2024
## Date of revision: 11/23/2024
#
## Project: MLGenie
## Description: This file defines the classes of data processing.
#
###############################################################
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from .Data import DataBase

class TransformerBase(ABC):
    """
    Base class for data transformers.
    """

    @abstractmethod
    def fit(self, data: DataBase, layer_in):
        """
        Fit the transformer on the input data.

        Parameters:
            data (DataBase): The DataBase object containing the feature matrix and metadata.
            layer_in (str): The layer from which to get the features.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, data: DataBase, layer_in, layer_out):
        """
        Transform the input data.

        Parameters:
            data (DataBase): The DataBase object containing the feature matrix and metadata.
            layer_in (str): The layer from which to get the features.
            layer_out (str): The layer to store the processed features.

        Returns:
            DataBase: Transformed data.
        """
        raise NotImplementedError

    def fit_transform(self, data: DataBase, layer_in="raw"):
        """
        Fit the transformer on the input data and transform the data in one step.

        Parameters:
            data (DataBase): The DataBase object containing the feature matrix and metadata.
            layer_in (str): The layer from which to get the features.

        Returns:
            DataBase: Transformed data.
        """
        if data is None or not isinstance(data, DataBase):
            raise TypeError("data must be a DataBase object.")
        if layer_in not in ["raw", "processed"]:
            raise ValueError("layer_in must be 'raw' or 'processed'.")
        self.fit(data, layer_in=layer_in)
        return self.transform(data, layer_in=layer_in, layer_out="processed")


class DataProcessorBase(object):
    """
    Base class for data processors.
    """

    def __init__(
        self, null_ratio_threshold=0.5, impute_strategy="mean", impute_fill_value=None,
    ):
        """
        Params:
            --- high quality feature param ---
            null_ratio_threshold (float): The maximum allowed percentage of missing values in a feature.
            --- impute missing value param ---
            impute_strategy (str): The imputation strategy for features. Options are 'mean', 'median', most_frequent' or 'constant'.
            impute_fill_value (float): The constant value to impute for missing values when strategy == 'constant'
        """
        self.null_ratio_threshold = null_ratio_threshold
        self.impute_strategy = impute_strategy
        self.impute_fill_value = impute_fill_value

        # Validate all parameters.
        self._validate_params()

        # Initialize processors
        self.high_quality_feature_filter = HighQualityFeatureFilter(
            null_ratio_threshold=self.null_ratio_threshold,
        )

        self.missing_value_imputer = MissingValueImputer(
            impute_strategy=self.impute_strategy,
            impute_fill_value=self.impute_fill_value,
        )

        self.processors: List = [
            self.high_quality_feature_filter,
            self.missing_value_imputer,
        ]

    def _validate_params(self):
        """
        Validate the base parameters.
        """
        # Validate null_ratio_threshold.
        if self.null_ratio_threshold is not None:
            if not isinstance(self.null_ratio_threshold, (int, float)):
                raise TypeError("null_ratio_threshold must be a number or None.")
            if not (0 <= self.null_ratio_threshold < 1):
                raise ValueError(
                    "null_ratio_threshold must be between 0 and 1 (not including 1) or None."
                )

        # Validate impute_strategy.
        valid_impute_strategies = ["mean", "median", "most_frequent", "constant"]
        if self.impute_strategy not in valid_impute_strategies:
            raise ValueError(
                f"Invalid impute_strategy. Expected one of: {valid_impute_strategies}"
            )
        # Validate impute_fill_value.
        if self.impute_strategy == "constant" and self.impute_fill_value is None:
            raise ValueError(
                "impute_fill_value must be specified when impute_strategy is 'constant'."
            )
        if self.impute_fill_value is not None and not isinstance(
            self.impute_fill_value, (int, float)
        ):
            raise TypeError("impute_fill_value must be a number or None.")

    def fit(self, data: DataBase):
        """
        Fit the data processor on the input DataBase by applying each processor in sequence.

        Parameters:
            data (DataBase): The DataBase object containing the feature matrix and metadata.
        """
        # Start with the raw layer
        current_layer = "raw"

        for processor in self.processors:
            data = processor.fit_transform(
                data, layer_in=current_layer
            )
            current_layer = "processed"

        return self

    def transform(self, data: DataBase) -> DataBase:
        """
        Transform the data by applying each processor in sequence.

        Parameters:
            data (DataBase): The DataBase object containing the feature matrix and metadata.

        Returns:
            DataBase: Transformed data.
        """
        current_layer = "raw"

        for processor in self.processors:
            if isinstance(processor, OutlierCorrector):
                data = processor.fit_transform(
                    data, layer_in=current_layer
                )
            else:
                data = processor.transform(
                    data, layer_in=current_layer, layer_out="processed"
                )
            current_layer = "processed"

        return data

    def fit_transform(self, data: DataBase):
        """
        Fit the data processor on the input data and transform the data in one step.

        Parameters:
            data (DataBase): The DataBase object containing the feature matrix and metadata.

        Returns:
            DataBase: Transformed data.
        """
        if data is None or not isinstance(data, DataBase):
            raise TypeError("data must be a DataBase object.")
        return self.fit(data).transform(data)


class BinaryDataProcessor(DataProcessorBase):
    """
    Processor for binary data.
    """

    def __init__(
        self,
        null_ratio_threshold=0.5,
        dominance_threshold=0.95,
        impute_strategy="most_frequent",
        impute_fill_value=0,
        chi2_p_threshold=0.5,
    ):
        """
        Params:
            --- high quality feature param ---
            null_ratio_threshold (float): The maximum allowed percentage of missing values in a feature.
            dominance_threshold (float): The maximum allowed percentage of the dominant class in binary features.
            --- informative feature param ---
            chi2_p_threshold (float): The maximum allowed p value of the Chi-square test for binary features with a binary target variable.
            --- impute missing value param ---
            impute_strategy (str): The imputation strategy for features. Options are 'most_frequent' or 'constant'.
            impute_fill_value (float): The constant value to impute for missing values of binary features when strategy == 'constant'
        """
        self.dominance_threshold = dominance_threshold
        self.chi2_p_threshold = chi2_p_threshold
        super().__init__(
            null_ratio_threshold=null_ratio_threshold,
            impute_strategy=impute_strategy,
            impute_fill_value=impute_fill_value,
        )

        # Initialize processors
        self.high_quality_feature_filter.dominance_threshold = self.dominance_threshold
        self.informative_feature_filter = InformativeFeatureFilter(
            chi2_p_threshold=self.chi2_p_threshold,
        )
        self.feature_scaler = FeatureScaler(
            scale_method="standard"
        )
        self.processors = [
            self.high_quality_feature_filter,
            self.informative_feature_filter,
            self.missing_value_imputer,
            self.feature_scaler
        ]

    def _validate_params(self,):
        """
        Validate the parameters.
        """
        super()._validate_params()

        # Validate dominance_threshold.
        if self.dominance_threshold is not None:
            if not isinstance(self.dominance_threshold, (int, float)):
                raise TypeError("dominance_threshold must be a number or None.")
            if not (0.5 <= self.dominance_threshold <= 1):
                raise ValueError(
                    "dominance_threshold must be between 0.5 and 1 or None."
                )

        # Validate chi2_p_threshold.
        if self.chi2_p_threshold is not None:
            if not isinstance(self.chi2_p_threshold, (int, float)):
                raise TypeError("chi2_p_threshold must be a number or None.")
            if not (0 <= self.chi2_p_threshold <= 1):
                raise ValueError("chi2_p_threshold must be between 0 and 1 or None.")

        # Validate impute_strategy.
        valid_impute_strategies = ["most_frequent", "constant"]
        if self.impute_strategy not in valid_impute_strategies:
            raise ValueError(
                f"Invalid impute_strategy. Expected one of: {valid_impute_strategies}"
            )


class ContinuesDataProcessor(DataProcessorBase):
    """
    Processor for continues data.
    """

    def __init__(
        self,
        null_ratio_threshold=0.5,
        variance_threshold=0.01,
        impute_strategy="mean",
        impute_fill_value=0,
        fold_change_threshold=None,
        target_corr_threshold=0,
        skew_threshold=2.5,
        skew_correction_method="yeo-johnson",
        outlier_detection_method="z_score",
        outlier_detection_threshold=3,
        outlier_correction_method="clip",
        scaling_method="minmax",
        scaling_feature_range=(0, 1),
        feature_corr_threshold=0.95,
    ):
        """
        Params:
            --- high quality feature param ---
            null_ratio_threshold (float): The maximum allowed percentage of missing values in a feature.
            variance_threshold (float): The minimum required variance for continuous features.
            --- impute missing value param ---
            impute_strategy (str): The imputation strategy for features. Options are 'mean', 'median' or 'constant'.
            impute_fill_value (float): The constant value to impute for missing values of continuous features when strategy == 'constant'
            --- informative feature param ---
            fold_change_threshold (float): The minimum fold change required for continuous features with a binary target variable.
            target_corr_threshold (float): The minimum correlation required for features with a continuous target variable.
            --- skewness param ---
            skew_threshold (float): The skewness threshold to identify skewed features.
            skew_correction_method (str): The method to use for correcting skewness.
            --- outlier param ---
            outlier_detection_method (str): The method to use for detecting outliers. Options are 'z_score', 'iqr', or 'modified_z_score'.
            outlier_detection_threshold (float): The threshold to use for detecting outliers.
            outlier_correction_method (str): The method to use for correcting outliers. Options are 'clip', 'mean' or 'median'.
            --- scaling param ---
            scaling_method (str):  The method to use for scaling outliers. Options are 'standard', 'minmax' or 'robust'.
            scaling_feature_range (tuple): The feature range to scale to.
            --- correlation param ---
            feature_corr_threshold (float): The correlation coefficient threshold above which features are considered highly correlated.
        """
        self.variance_threshold = variance_threshold
        self.fold_change_threshold = fold_change_threshold
        self.target_corr_threshold = target_corr_threshold
        self.skew_threshold = skew_threshold
        self.skew_correction_method = skew_correction_method
        self.outlier_detection_method = outlier_detection_method
        self.outlier_detection_threshold = outlier_detection_threshold
        self.outlier_correction_method = outlier_correction_method
        self.scaling_method = scaling_method
        self.scaling_feature_range = scaling_feature_range
        self.feature_corr_threshold = feature_corr_threshold

        super().__init__(
            null_ratio_threshold=null_ratio_threshold,
            impute_strategy=impute_strategy,
            impute_fill_value=impute_fill_value,
        )

        # Initialize processors
        self.high_quality_feature_filter.variance_threshold = self.variance_threshold
        self.informative_feature_filter = InformativeFeatureFilter(
            target_corr_threshold=self.target_corr_threshold,
            fold_change_threshold=self.fold_change_threshold,
        )
        self.skewness_corrector = SkewnessCorrector(
            skew_threshold=self.skew_threshold,
            skew_correction_method=self.skew_correction_method,
        )
        self.outlier_corrector = OutlierCorrector(
            detection_method=self.outlier_detection_method,
            detection_threshold=self.outlier_detection_threshold,
            correction_method=self.outlier_correction_method,
        )
        self.feature_scaler = FeatureScaler(
            scale_method=self.scaling_method, feature_range=self.scaling_feature_range
        )
        self.corrected_feature_filter = CorrelatedFeatureFilter(
            feature_corr_threshold=self.feature_corr_threshold
        )

        self.processors = [
            self.high_quality_feature_filter,
            self.informative_feature_filter,
            self.missing_value_imputer,
            self.skewness_corrector,
            self.outlier_corrector,
            self.feature_scaler,
            self.corrected_feature_filter,
        ]

    def _validate_params(self,):
        super()._validate_params()

        # Validate variance_threshold.
        if self.variance_threshold is not None:
            if not isinstance(self.variance_threshold, (int, float)):
                raise TypeError("variance_threshold must be a number or None.")
            if self.variance_threshold < 0:
                raise ValueError("variance_threshold must be non-negative or None.")

        # Validate impute_strategy.
        valid_impute_strategies = ["mean", "median", "constant"]
        if self.impute_strategy not in valid_impute_strategies:
            raise ValueError(
                f"Invalid impute_strategy. Expected one of: {valid_impute_strategies}"
            )

        # Validate fold_change_threshold.
        if self.fold_change_threshold is not None:
            if not isinstance(self.fold_change_threshold, (int, float)):
                raise TypeError("fold_change_threshold must be a number or None.")
            if self.fold_change_threshold < 1:
                raise ValueError(
                    "fold_change_threshold must be greater than 1 or None."
                )

        # Validate target_corr_threshold.
        if self.target_corr_threshold is not None:
            if not isinstance(self.target_corr_threshold, (int, float)):
                raise TypeError("target_corr_threshold must be a number or None.")
            if not (0 <= self.target_corr_threshold <= 1):
                raise ValueError(
                    "target_corr_threshold must be between 0 and 1 or None."
                )

        # Validate skew_threshold.
        if self.skew_threshold is not None:
            if not isinstance(self.skew_threshold, (int, float)):
                raise TypeError("skew_threshold must be a number or None.")
            if self.skew_threshold <= 0:
                raise ValueError("skew_threshold must be greater than 0.")

        # Validate skew_correction_method.
        valid_skew_correction_methods = ["exp", "square", "box-cox", "yeo-johnson"]
        if self.skew_correction_method not in valid_skew_correction_methods:
            raise ValueError(
                f"Invalid skew_correction_method. Expected one of: {valid_skew_correction_methods}"
            )

        # Validate outlier_detection_method.
        valid_outlier_detection_methods = ["z_score", "iqr", "modified_z_score"]
        if self.outlier_detection_method not in valid_outlier_detection_methods:
            raise ValueError(
                f"Invalid outlier_detection_method. Expected one of: {valid_outlier_detection_methods}"
            )
        # Validate outlier_detection_threshold.
        if not isinstance(self.outlier_detection_threshold, (int, float)):
            raise TypeError("detection_threshold must be a number.")
        if self.outlier_detection_threshold < 0:
            raise ValueError("detection_threshold must be non-negative.")

        # Validate outlier_correction_method.
        valid_outlier_correction_methods = ["clip", "mean", "median"]
        if self.outlier_correction_method not in valid_outlier_correction_methods:
            raise ValueError(
                f"Invalid outlier_correction_method. Expected one of: {valid_outlier_correction_methods}"
            )

        # Validate scaling_method.
        valid_scaling_methods = ["standard", "minmax", "robust"]
        if self.scaling_method not in valid_scaling_methods:
            raise ValueError(
                f"Invalid scaling_method. Expected one of: {valid_scaling_methods}"
            )
        if self.scaling_feature_range is not None:
            if not all(
                isinstance(val, (int, float)) for val in self.scaling_feature_range
            ):
                raise TypeError("feature_range values must be numbers.")
            if self.scaling_feature_range[0] >= self.scaling_feature_range[1]:
                raise ValueError("feature_range must be in increasing order.")

        # Validate feature_corr_threshold.
        if self.feature_corr_threshold is not None:
            if not isinstance(self.feature_corr_threshold, (int, float)):
                raise TypeError("feature_corr_threshold must be a number or None.")
            if not (0 <= self.feature_corr_threshold <= 1):
                raise ValueError(
                    "feature_corr_threshold must be between 0 and 1 or None."
                )


class HighQualityFeatureFilter(TransformerBase):
    """
    Identify features based on null ratio, dominance, and variance thresholds.
    """

    def __init__(
        self,
        null_ratio_threshold=None,
        dominance_threshold=None,
        variance_threshold=None,
    ):
        """
        Parameters:
            null_ratio_threshold (float): The maximum allowed percentage of missing values in a feature.
            dominance_threshold (float): The maximum allowed percentage of the dominant class in binary features.
            variance_threshold (float): The minimum required variance for continuous features.
        """
        if null_ratio_threshold is not None:
            if not isinstance(null_ratio_threshold, (int, float)):
                raise TypeError("null_ratio_threshold must be a number or None.")
            if not (0 <= null_ratio_threshold < 1):
                raise ValueError(
                    "null_ratio_threshold must be between 0 and 1 (not including 1) or None."
                )
        if dominance_threshold is not None:
            if not isinstance(dominance_threshold, (int, float)):
                raise TypeError("dominance_threshold must be a number or None.")
            if not (0.5 <= dominance_threshold <= 1):
                raise ValueError(
                    "dominance_threshold must be between 0.5 and 1 or None."
                )
        if variance_threshold is not None:
            if not isinstance(variance_threshold, (int, float)):
                raise TypeError("variance_threshold must be a number or None.")
            if variance_threshold < 0:
                raise ValueError("variance_threshold must be non-negative or None.")

        self.null_ratio_threshold = null_ratio_threshold
        self.dominance_threshold = dominance_threshold
        self.variance_threshold = variance_threshold
        self.selected_features_ = None

    def fit(self, data: DataBase, layer_in="raw"):
        """
        Fit the filter on input DataBase. Identify high-quality features based on thresholds.

        Parameters:
            data (DataBase): The DataBase object containing the feature matrix and metadata.
            layer_in (str): The layer from which to get the features.
        """
        if data is None or not isinstance(data, DataBase):
            raise TypeError("data must be a DataBase object.")
        if layer_in not in ["raw", "processed"]:
            raise ValueError("layer_in must be 'raw' or 'processed'.")

        X = data.get_features(layer=layer_in)
        self.selected_features_ = self._identify_high_quality_features(X)

        # Check if any features meet the criteria.
        if self.selected_features_.empty:
            raise RuntimeError(
                "No features meet the criteria for high quality. Please adjust the thresholds."
            )
        return self

    def transform(
        self, data: DataBase, layer_in="raw", layer_out="processed"
    ) -> DataBase:
        """
        Transform the DataBase object by selecting high-quality features based on the fit.

        Parameters:
            data (DataBase): The DataBase object containing the feature matrix and metadata.
            layer_in (str): The layer from which to get the features.
            layer_out (str): The layer to store the processed features.

        Returns:
            DataBase: Transformed data with high-quality features selected.
        """
        if self.selected_features_ is None:
            raise ValueError("The estimator must be fit before transforming the data.")
        if not isinstance(data, DataBase):
            raise TypeError("data must be a DataBase object.")
        if layer_in not in ["raw", "processed"]:
            raise ValueError("layer_in must be 'raw' or 'processed'.")
        if layer_out not in ["raw", "processed"]:
            raise ValueError("layer_out must be 'raw' or 'processed'.")

        X = data.get_features(layer=layer_in)
        if X is None or not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if not all(col in X.columns for col in self.selected_features_):
            print(X.columns)
            print(self.selected_features_)
            raise ValueError(
                "The selected features must be a subset of the features in the input data."
            )
        X_transformed = X[self.selected_features_]
        data.update_features(X_transformed, layer=layer_out)
        return data

    def _identify_high_quality_features(self, X: pd.DataFrame) -> pd.Index:
        """
        Identify features that are considered high quality based on the following criteria.

        - Features with a higher percentage of missing values than `null_ratio_threshold` will be removed.
        - Binary features with a higher percentage of the dominant class than `dominance_threshold` will be removed.
        - Continuous features with a variance below `variance_threshold` will be removed.

        Parameters:
            X (pd.DataFrame): The feature matrix with feature names.
        
        Returns:
            pd.Index: Index of the selected features.
        """
        if X is None or not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        print("Identifying high quality features ...", flush=True)

        # Initialize boolean arrays for feature selection
        keep_null = pd.Series(True, index=X.columns)
        keep_variance = pd.Series(True, index=X.columns)
        keep_dominance = pd.Series(True, index=X.columns)

        if self.null_ratio_threshold is not None:
            # Calculate null ratios
            null_ratios = X.isnull().mean(axis=0)
            keep_null = null_ratios <= self.null_ratio_threshold

        if self.variance_threshold is not None:
            # Calculate variance
            variances = X.var(axis=0, skipna=True)
            keep_variance = variances >= self.variance_threshold

        if self.dominance_threshold is not None:
            keep_dominance = pd.Series(True, index=X.columns)

            # Identify binary features and calculate dominance
            binary_features = X.nunique() <= 2
            if binary_features.any():
                dominance = X.loc[:, binary_features].apply(
                    lambda x: x.value_counts(normalize=True).max(), axis=0
                )
                keep_dominance[binary_features] = dominance <= self.dominance_threshold

        # Combine all boolean arrays to get final selected features
        keep_final = keep_null & keep_variance & keep_dominance

        return keep_final.index[keep_final]


class InformativeFeatureFilter(TransformerBase):
    """
    Identify informative features based on target variable type and feature type.
    """

    def __init__(
        self,
        target_corr_threshold=None,
        chi2_p_threshold=None,
        fold_change_threshold=None,
    ):
        """
        Parameters:
            target_corr_threshold (float): The minimum correlation required for features with a continuous target variable.
            chi2_p_threshold (float): The maximum allowed p value of the Chi-square test for binary features with a binary target variable.
            fold_change_threshold (float): The minimum fold change required for continuous features with a binary target variable.
        """
        if target_corr_threshold is not None:
            if not isinstance(target_corr_threshold, (int, float)):
                raise TypeError("target_corr_threshold must be a number or None.")
            if not (0 <= target_corr_threshold <= 1):
                raise ValueError(
                    "target_corr_threshold must be between 0 and 1 or None."
                )
        if chi2_p_threshold is not None:
            if not isinstance(chi2_p_threshold, (int, float)):
                raise TypeError("chi2_p_threshold must be a number or None.")
            if not (0 <= chi2_p_threshold <= 1):
                raise ValueError("chi2_p_threshold must be between 0 and 1 or None.")
        if fold_change_threshold is not None:
            if not isinstance(fold_change_threshold, (int, float)):
                raise TypeError("fold_change_threshold must be a number or None.")
            if fold_change_threshold < 1:
                raise ValueError(
                    "fold_change_threshold must be greater than 1 or None."
                )

        self.target_corr_threshold = target_corr_threshold
        self.chi2_p_threshold = chi2_p_threshold
        self.fold_change_threshold = fold_change_threshold
        self.identified_features_ = None

    def fit(self, data: DataBase, layer_in="raw"):
        """
        Fit the feature filter on input DataBase. Identify informative features based on thresholds.

        Parameters:
            data (DataBase): The DataBase object containing the feature matrix and metadata.
            layer_in (str): The layer from which to get the features.
        """
        if data is None or not isinstance(data, DataBase):
            raise TypeError("data must be a DataBase object.")
        if layer_in not in ["raw", "processed"]:
            raise ValueError("layer_in must be 'raw' or 'processed'.")

        X = data.get_features(layer=layer_in)
        y = data.get_labels()
        self.identified_features_ = self._identify_informative_features(X, y)

        # Check if any features meet the criteria.
        if self.identified_features_.empty:
            raise RuntimeError(
                "No features meet the criteria for informative features. Please adjust the thresholds."
            )
        return self

    def transform(
        self, data: DataBase, layer_in="raw", layer_out="processed"
    ) -> DataBase:
        """
        Transform the DataBase object by applying the informative feature selection.

        Parameters:
            data (DataBase): The DataBase object containing the feature matrix and metadata.
            layer_in (str): The layer from which to get the features.
            layer_out (str): The layer to store the processed features.

        Returns:
            DataBase: Transformed data containing only informative features.
        """
        if self.identified_features_ is None:
            raise ValueError("The estimator must be fit before transforming the data.")
        if data is None or not isinstance(data, DataBase):
            raise TypeError("data must be a DataBase object.")
        if layer_in not in ["raw", "processed"]:
            raise ValueError("layer_in must be 'raw' or 'processed'.")
        if layer_in not in ["raw", "processed"]:
            raise ValueError("layer_out must be 'raw' or 'processed'.")

        X = data.get_features(layer=layer_in)
        if X is None or not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        for col in self.identified_features_:
            if col not in X.columns:
                raise ValueError(
                    "The selected features must be a subset of the features in the input data."
                )
        X_transformed = X[self.identified_features_]
        data.update_features(X_transformed, layer=layer_out)
        return data

    def _identify_informative_features(self, X: pd.DataFrame, y: pd.Series):
        """
        Identify features that are considered informative based on the target variable type and feature type.

        - For binary features with a binary target variable:
            Filter out features with a p-value above `chi2_p_threshold` using Chi-square test.
        - For continuous features with a binary target variable:
            Filter out features with a fold change smaller than `fold_change_threshold`.
        - For features with a continuous target variable:
            Filter out features with a correlation below `target_corr_threshold`.

        Parameters:
            X (pd.DataFrame): The feature matrix with feature names.
            y (pd.Series): The target variable.
        
        Returns:
            pd.Index: Index of the selected features.
        """
        if X is None or not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if y is None or not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        print("Identifying informative features ...", flush=True)

        binary_target = y.nunique() == 2
        y_values = y.values

        if binary_target:
            return self._identify_binary_target(X, y_values)
        else:
            return self._identify_continuous_target(X, y)

    def _identify_binary_target(self, X, y):
        binary_features = X.nunique() == 2
        continuous_features = ~binary_features

        # Process binary features
        if binary_features.any():
            binary_results = self._process_binary_features(X.loc[:, binary_features], y)
        else:
            binary_results = pd.Series(dtype=bool)

        # Process continuous features
        if continuous_features.any():
            continuous_results = self._process_continuous_features(
                X.loc[:, continuous_features], y
            )
        else:
            continuous_results = pd.Series(dtype=bool)

        keep_features = pd.Series(False, index=X.columns)
        keep_features[binary_features] = binary_results
        keep_features[continuous_features] = continuous_results

        return keep_features.index[keep_features]

    def _identify_continuous_target(self, X, y):
        numeric_features = X.select_dtypes(include=[np.number]).columns
        keep_features = pd.Series(False, index=X.columns)
        if self.target_corr_threshold:
            correlations = X[numeric_features].corrwith(y)
            keep_features[numeric_features] = (
                np.abs(correlations) >= self.target_corr_threshold
            )
        else:
            keep_features[numeric_features] = True

        return keep_features.index[keep_features]

    def _process_binary_features(self, X, y):

        if self.chi2_p_threshold:
            X_encoded = X.apply(LabelEncoder().fit_transform)
            chi2_stats, p_values = chi2(X_encoded, y)
            return p_values <= self.chi2_p_threshold
        else:
            return np.ones(X.shape[1], dtype=bool)

    def _process_continuous_features(self, X, y):
        if X.empty:
            return pd.Series(dtype=bool)

        # Calculate means for each group
        group_means = np.array([X[y == label].mean() for label in np.unique(y)])

        # Calculate fold changes
        fold_changes = np.maximum(
            np.abs(group_means[0] / group_means[1]),
            np.abs(group_means[1] / group_means[0]),
        )

        # Apply threshold
        if self.fold_change_threshold:
            return fold_changes >= self.fold_change_threshold
        else:
            return np.ones(len(fold_changes), dtype=bool)

    # def _compute_correlation(self, X, y):
    #     """
    #     Compute correlation between features and target variable.
    #
    #     Parameters:
    #     X (pd.DataFrame): DataFrame of numeric features
    #     y (np.array): Target variable
    #
    #     Returns:
    #     np.array: Correlation coefficients for each feature
    #     """
    #     # Center the data
    #     X_centered = X - X.mean()
    #     y_centered = y - y.mean()
    #     mask = ~np.isnan(X_centered) & ~np.isnan(X_centered)
    #     print(mask)
    #
    #     # Compute correlation
    #     corr_numerator = np.dot(X_centered.T, y_centered)
    #     print(corr_numerator)
    #     corr_denominator = np.sqrt(
    #         np.sum(X_centered ** 2, axis=0) * np.sum(y_centered ** 2)
    #     )
    #     print(corr_denominator)
    #
    #     return corr_numerator / corr_denominator


class MissingValueImputer(TransformerBase):
    """
    Imputes missing values in the dataset using specified strategies for continuous and binary features.
    """

    def __init__(
        self, impute_strategy="mean", impute_fill_value=None,
    ):
        """
        Parameters:
            impute_strategy (str): The imputation strategy for features. Options are 'mean', 'median', most_frequent' or 'constant'.
            impute_fill_value (float): The constant value to impute for missing values when strategy == 'constant'
        """
        valid_impute_strategies = ["mean", "median", "most_frequent", "constant"]
        if impute_strategy not in valid_impute_strategies:
            raise ValueError(
                f"Invalid impute_strategy. Expected one of: {valid_impute_strategies}"
            )
        if impute_strategy == "constant" and impute_fill_value is None:
            raise ValueError(
                "impute_fill_value must be specified when impute_strategy is 'constant'."
            )
        if impute_fill_value is not None and not isinstance(
            impute_fill_value, (int, float)
        ):
            raise TypeError("impute_fill_value must be a number or None.")

        self.impute_strategy = impute_strategy
        self.impute_fill_value = impute_fill_value
        self.impute_values_ = None

    def fit(self, data: DataBase, layer_in="raw"):
        """
        Fit the imputer on the input DataBase. Calculates imputation values for continuous and binary features.

        Parameters:
            data (DataBase): The DataBase object containing the feature matrix and metadata.
            layer_in (str): The layer from which to get the features.
        """
        if data is None or not isinstance(data, DataBase):
            raise TypeError("data must be a DataBase object.")
        X = data.get_features(layer=layer_in)
        self.impute_values_ = self._impute_missing_values(X)
        return self

    def transform(self, data: DataBase, layer_in="raw", layer_out="processed"):
        """
        Transform the DataBase object by imputing missing values based on the fit imputer.

        Parameters:
            data (DataBase): The DataBase object containing the feature matrix and metadata.
            layer_in (str): The layer from which to get the features.
            layer_out (str): The layer to store the processed features.
        """
        if self.impute_values_ is None:
            raise ValueError("The estimator must be fit before transforming the data.")
        if data is None or not isinstance(data, DataBase):
            raise TypeError("data must be a DataBase object.")
        if layer_in not in ["raw", "processed"]:
            raise ValueError("layer_in must be 'raw' or 'processed'.")
        if data.get_features(layer=layer_in) is None:
            raise ValueError("The input data must contain features.")

        X = data.get_features(layer=layer_in).copy()
        # Validate that the new data has the same columns as the data used to fit the imputer
        if not all(col in X.columns for col in self.impute_values_):
            raise ValueError(
                "New data must contain the same features as the data used to fit the imputer."
            )

        X.fillna(self.impute_values_, inplace=True)

        data.update_features(X, layer=layer_out)
        return data

    def _impute_missing_values(self, X: pd.DataFrame) -> dict:
        """
        Imputes missing values in the dataset using specified strategies for continuous and binary features.

        - If `impute_strategy_continuous` is 'mean', replace missing values with the mean of the feature.
        - If `impute_strategy_continuous` is 'median', replace missing values with the median of the feature.
        - If `impute_strategy_binary` is 'most_frequent', replace missing values with the mode of the feature.
        - If `impute_strategy_binary` is 'constant', replace missing values with `impute_fill_value`.
        """
        if X is None or not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        print("Imputing missing value ...", flush=True)

        impute_values = {}
        # Precompute impute values based on the strategy to avoid redundant calculations
        if self.impute_strategy == "mean":
            impute_values = X.mean().to_dict()
        elif self.impute_strategy == "median":
            impute_values = X.apply(
                lambda col: col.median() if col.nunique() > 0 else np.nan, axis=0
            ).to_dict()
        elif self.impute_strategy == "most_frequent":
            impute_values = X.apply(
                lambda col: col.mode().iloc[0] if col.nunique() > 0 else np.nan, axis=0
            ).to_dict()
        elif self.impute_strategy == "constant":
            impute_values = {col: self.impute_fill_value for col in X.columns}
        return impute_values


class SkewnessCorrector(TransformerBase):
    """
    Corrects skewness in numerical features using specified methods.
    """

    def __init__(self, skew_threshold=0.5, skew_correction_method="box-cox"):
        """
        Parameters:
            skew_threshold (float): The skewness threshold to identify skewed features.
            skew_correction_method (str): The method to use for correcting skewness. Options are 'exp', 'square', 'box-cox', and 'yeo-johnson'.
        """
        valid_methods = ["exp", "square", "box-cox", "yeo-johnson"]

        if skew_correction_method not in valid_methods:
            raise ValueError(
                f"Invalid skew_correction_method. Expected one of: {valid_methods}"
            )

        if skew_threshold is not None:
            if not isinstance(skew_threshold, (int, float)):
                raise TypeError("skew_threshold must be a number.")
            if skew_threshold <= 0:
                raise ValueError("The skewness threshold must be greater than 0.")

        self.skew_threshold = skew_threshold
        self.skew_correction_method = skew_correction_method
        self.left_skewed_features_ = None
        self.right_skewed_features_ = None

    def fit(self, data: DataBase, layer_in="raw"):
        """
        Fit the skewness corrector on input DataBase. Identify skewed features based on the skewness threshold.

        Parameters:
            data (DataBase): The DataBase object containing the feature matrix and metadata.
            layer_in (str): The layer from which to get the features.
        """
        if data is None or not isinstance(data, DataBase):
            raise TypeError("data must be a DataBase object.")
        if layer_in not in ["raw", "processed"]:
            raise ValueError("layer_in must be 'raw' or 'processed'.")
        X = data.get_features(layer=layer_in)
        (
            self.left_skewed_features_,
            self.right_skewed_features_,
        ) = self._identify_skewed_features(X)
        return self

    def transform(
        self, data: DataBase, layer_in="raw", layer_out="processed"
    ) -> DataBase:
        """
        Transform the DataBase object by applying skewness correction to identified skewed features.

        Parameters:
            data (DataBase): The DataBase object containing the feature matrix and metadata.
            layer_in (str): The layer from which to get the features.
            layer_out (str): The layer to store the processed features.

        Returns:
            DataBase: Transformed data with corrected skewed features.
        """
        if self.left_skewed_features_ is None or self.right_skewed_features_ is None:
            raise ValueError("The estimator must be fit before transforming the data.")
        if data is None or not isinstance(data, DataBase):
            raise TypeError("data must be a DataBase object.")
        if layer_in not in ["raw", "processed"]:
            raise ValueError("layer_in must be 'raw' or 'processed'.")
        if layer_out not in ["raw", "processed"]:
            raise ValueError("layer_out must be 'raw' or 'processed'.")
        if data.get_features(layer=layer_in) is None:
            raise ValueError("The input data must contain features.")

        X = data.get_features(layer=layer_in).copy()
        skewed_features = self.left_skewed_features_ + self.right_skewed_features_

        if any(feature not in X.columns for feature in skewed_features):
            raise ValueError(
                "Selected features must be a subset of input data features."
            )

        # Apply skewness correction in a vectorized way
        X[self.left_skewed_features_] = np.log(X[self.left_skewed_features_])
        X[self.right_skewed_features_] = np.clip(
            np.exp(X[self.right_skewed_features_]), None, np.exp(5)
        )

        data.update_features(X, layer=layer_out)
        return data

    def _identify_skewed_features(self, X: pd.DataFrame) -> Tuple[List, List]:
        """
        Identify features with skewness above the specified threshold.

        Parameters:
            X (pd.DataFrame): The feature matrix with feature names.
        
        Returns:
            left_skewed_features (list): List of left skewed feature names.
            right_skewed_features (list): List of right skewed feature names.
        """
        if X is None or not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if self.skew_threshold is None:
            return [], []
        else:
            print("Identifying and transforming skewed features ...", flush=True)
            numeric_X = X.select_dtypes(include=[np.number])
            positive_X = numeric_X.loc[:, (numeric_X > 0).all()]
            skewness = positive_X.skew()
            left_skewed_features = skewness[skewness < -self.skew_threshold].index.tolist()
            right_skewed_features = skewness[skewness > self.skew_threshold].index.tolist()

            return left_skewed_features, right_skewed_features


class OutlierCorrector(TransformerBase):
    """
    Detects and corrects outliers in numerical features using specified methods.
    """

    def __init__(
        self,
        detection_method="z_score",
        detection_threshold=3,
        correction_method="clip",
    ):
        """
        Parameters:
            detection_method (str): The method to use for detecting outliers. Options are 'z_score', 'iqr', and 'modified_z_score'.
            detection_threshold (float): The threshold to use for detecting outliers.
            correction_method (str): The method to use for correcting outliers. Options are 'clip', 'mean', and 'median'.
        """
        valid_detection_methods = ["z_score", "iqr", "modified_z_score"]
        valid_correction_methods = ["clip", "mean", "median"]

        if detection_method not in valid_detection_methods:
            raise ValueError(
                f"Invalid detection_method. Expected one of: {valid_detection_methods}"
            )
        if correction_method not in valid_correction_methods:
            raise ValueError(
                f"Invalid correction_method. Expected one of: {valid_correction_methods}"
            )
        if not isinstance(detection_threshold, (int, float)):
            raise TypeError("detection_threshold must be a number.")
        if detection_method == "z_score" and detection_threshold <= 0:
            raise ValueError(
                "The detection threshold must be greater than 0 when using z_score method."
            )
        if detection_method == "modified_z_score" and detection_threshold <= 0:
            raise ValueError(
                "The detection threshold must be greater than 0 when using modified_z_score method."
            )
        if detection_method == "iqr" and detection_threshold <= 1:
            raise ValueError(
                "The detection threshold must be greater than 1 when using iqr method."
            )

        self.detection_method = detection_method
        self.detection_threshold = detection_threshold
        self.correction_method = correction_method
        self.outliers_ = None

    def fit(self, data: DataBase, layer_in: str = "raw"):
        """
        Fit the outlier corrector on input DataBase. Identify outliers based on the detection method and threshold.

        Parameters:
            data (DataBase): The DataBase object containing the feature matrix and metadata.
            layer_in (str): The layer from which to get the features.
        """
        if data is None or not isinstance(data, DataBase):
            raise TypeError("data must be a DataBase object.")
        if layer_in not in ["raw", "processed"]:
            raise ValueError("layer_in must be 'raw' or 'processed'.")

        X = data.get_features(layer=layer_in)
        self.outliers_ = self._identify_outliers(X)
        return self

    def transform(
        self, data: DataBase, layer_in: str = "raw", layer_out: str = "processed"
    ) -> DataBase:
        """
        Transform the DataBase object by applying outlier correction to identified outliers.

        Parameters:
            data (DataBase): The DataBase object containing the feature matrix and metadata.
            layer_in (str): The layer from which to get the features.
            layer_out (str): The layer to store the processed features.

        Returns:
            DataBase: Transformed data with corrected outliers.
        """
        if self.outliers_ is None:
            raise ValueError("The estimator must be fit before transforming the data.")
        if data is None or not isinstance(data, DataBase):
            raise TypeError("data must be a DataBase object.")
        if layer_in not in ["raw", "processed"]:
            raise ValueError("layer_in must be 'raw' or 'processed'.")
        if layer_out not in ["raw", "processed"]:
            raise ValueError("layer_out must be 'raw' or 'processed'.")
        if data.get_features(layer=layer_in) is None:
            raise ValueError("The input data must contain features.")

        X = data.get_features(layer=layer_in)
        features_to_correct = list(self.outliers_.keys())

        if not set(features_to_correct).issubset(set(X.columns)):
            missing_features = set(features_to_correct) - set(X.columns)
            raise ValueError(
                f"Features {missing_features} not found in the input data."
            )

        # Select only the features that need correction
        X_subset = X[features_to_correct]

        # Perform the correction in-place
        self._correct_outliers(X_subset, inplace=True)

        # Update the original DataFrame with the corrected values
        X.loc[:, features_to_correct] = X_subset

        data.update_features(X, layer=layer_out)
        return data

    def _identify_outliers(self, X: pd.DataFrame) -> dict:
        """
        Identify outliers in the dataset based on the specified detection method and threshold.

        Parameters:
            X (pd.DataFrame): The feature matrix with feature names.
        
        Returns:
            dict: Dictionary where keys are feature names and values are boolean arrays indicating outliers.
        """
        if X is None or not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        print("Identifying feature outliers ...", flush=True)

        outliers = {}
        numeric_columns = X.select_dtypes(include=[np.number]).columns

        if self.detection_method == "z_score":
            outliers = self._z_score_outliers(X[numeric_columns].values, self.detection_threshold)
        elif self.detection_method == "iqr":
            outliers = self._iqr_outliers(X[numeric_columns].values, self.detection_threshold)
        elif self.detection_method == "modified_z_score":
            outliers = self._modified_z_score_outliers(
                X[numeric_columns].values, self.detection_threshold
            )

        return {
            col: pd.Series(outliers[:, i], index=X.index)
            for i, col in enumerate(numeric_columns)
        }

    @staticmethod
    def _z_score_outliers(X, threshold):
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        z_scores = np.abs((X - mean) / std)
        return z_scores > threshold

    @staticmethod
    def _iqr_outliers(X, threshold):
        Q1 = np.nanpercentile(X, 25, axis=0)
        Q3 = np.nanpercentile(X, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (X < lower_bound) | (X > upper_bound)

    @staticmethod
    def _modified_z_score_outliers(X, threshold):
        median = np.median(X, axis=0)
        mad = np.median(np.abs(X - median), axis=0)
        modified_z_scores = 0.6745 * np.abs(X - median) / mad
        return modified_z_scores > threshold

    def _correct_outliers(
        self, features: pd.DataFrame, inplace: bool = False
    ) -> pd.DataFrame:
        """
        Corrects outliers in numerical features of the dataset using specified methods.
        """
        if features is None or not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame")
        if self.outliers_ is None:
            raise ValueError("The estimator must be fit before transforming the data.")
        if not all(col in self.outliers_.keys() for col in features.columns):
            raise ValueError(
                "All features to correct must have been identified as outliers."
            )

        if not inplace:
            features = features.copy()

        outliers_mask = pd.DataFrame(
            {col: self.outliers_[col] for col in features.columns}
        )

        if self.correction_method == "clip":
            if self.detection_method == "iqr":
                Q1 = features.quantile(0.25)
                Q3 = features.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.detection_threshold * IQR
                upper_bound = Q3 + self.detection_threshold * IQR
            else:
                mean = features.mean()
                std = features.std()
                lower_bound = mean - self.detection_threshold * std
                upper_bound = mean + self.detection_threshold * std

            features.clip(lower=lower_bound, upper=upper_bound, axis=1, inplace=True)
        elif self.correction_method == "mean":
            mean_values = features.mean()
            features.mask(outliers_mask, mean_values, inplace=True, axis=1)
        elif self.correction_method == "median":
            median_values = features.median(axis=0)
            features.mask(outliers_mask, median_values, inplace=True, axis=1)

        return features


class FeatureScaler(TransformerBase):
    """
    Scales numerical features using specified methods.
    """

    def __init__(self, scale_method="minmax", feature_range=(0, 1)):
        """
        Parameters:
            method (str): The scaling method to use. Options are 'standard', 'minmax', and 'robust'.
            feature_range (tuple): The range to scale features to when using 'minmax' scaling.
        """
        valid_methods = ["standard", "minmax", "robust"]

        if scale_method not in valid_methods:
            raise ValueError(f"Invalid method. Expected one of: {valid_methods}")
        if scale_method == "minmax" and not (
            isinstance(feature_range, tuple) and len(feature_range) == 2
        ):
            raise ValueError(
                "feature_range must be a tuple of two values for 'minmax' scaling."
            )
        if feature_range:
            if len(feature_range) != 2:
                raise ValueError("feature_range must be a tuple of two values.")
            if not all(isinstance(val, (int, float)) for val in feature_range):
                raise TypeError("feature_range values must be numbers.")
            if feature_range[0] >= feature_range[1]:
                raise ValueError("feature_range must be in increasing order.")

        self.scale_method = scale_method
        if scale_method == "minmax":
            self.scale_range = feature_range
        else:
            self.scale_range = None
        self.scaler_ = None

    def fit(self, data: DataBase, layer_in="raw"):
        """
        Fit the scaler on input DataBase.

        Parameters:
            data (DataBase): The DataBase object containing the feature matrix and metadata.
            layer_in (str): The layer from which to get the features.
        """
        if data is None or not isinstance(data, DataBase):
            raise TypeError("data must be a DataBase object.")
        if layer_in not in ["raw", "processed"]:
            raise ValueError("layer_in must be 'raw' or 'processed'.")

        X = data.get_features(layer=layer_in)
        if X is None or not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        self.scaler_ = self._select_scaler()
        print("Scaling features ...", flush=True)
        self.scaler_.fit(X)
        return self

    def transform(
        self, data: DataBase, layer_in="raw", layer_out="processed"
    ) -> DataBase:
        """
        Transform the DataBase object by applying the scaling to numerical features.

        Parameters:
            data (DataBase): The DataBase object containing the feature matrix and metadata.

        Returns:
            DataBase: Transformed data with scaled features.
        """
        if self.scaler_ is None:
            raise ValueError("The estimator must be fit before transforming the data.")
        if data is None or not isinstance(data, DataBase):
            raise TypeError("data must be a DataBase object.")
        if data.get_features(layer=layer_in) is None:
            raise ValueError("The input data must contain features.")

        X = data.get_features(layer=layer_in).copy()
        X_scaled = pd.DataFrame(
            self.scaler_.transform(X), columns=X.columns, index=X.index
        )

        data.update_features(X_scaled, layer=layer_out)
        return data

    def _select_scaler(self):
        """
        Selects the appropriate scaler based on the specified method.
        
        Returns:
            scaler: The initialized scaler object.
        """
        if self.scale_method == "standard":
            return StandardScaler()
        elif self.scale_method == "minmax":
            return MinMaxScaler(feature_range=self.scale_range)
        elif self.scale_method == "robust":
            return RobustScaler()


class CorrelatedFeatureFilter(TransformerBase):
    """
    Drops features with high correlation based on a specified correlation threshold.
    """

    def __init__(self, feature_corr_threshold=0.9):
        """
        Parameters:
            feature_corr_threshold (float): The correlation threshold above which features are considered highly correlated.
        """
        if feature_corr_threshold is not None:
            if not isinstance(feature_corr_threshold, (int, float)):
                raise TypeError("feature_corr_threshold must be a number.")
            if not (0 <= feature_corr_threshold <= 1):
                raise ValueError("feature_corr_threshold must be between 0 and 1.")

        self.feature_corr_threshold = feature_corr_threshold
        self.dropped_features_ = None

    def fit(self, data: DataBase, layer_in="raw"):
        """
        Fit the feature filter on input DataBase. Identify highly correlated features based on the threshold.

        Parameters:
            data (DataBase): The DataBase object containing the feature matrix and metadata.
            layer_in (str): The layer from which to get the features.
        """
        if data is None or not isinstance(data, DataBase):
            raise TypeError("data must be a DataBase object.")
        if layer_in not in ["raw", "processed"]:
            raise ValueError("layer_in must be 'raw' or 'processed'.")

        if not self.feature_corr_threshold:
            self.dropped_features_ = []
        else:
            X = data.get_features(layer=layer_in)
            y = data.get_labels()
            self.dropped_features_ = self._identify_highly_correlated_features(X, y)
        return self

    def transform(
        self, data: DataBase, layer_in: str = "raw", layer_out: str = "processed"
    ) -> DataBase:
        """
        Transform the DataBase object by dropping highly correlated features.

        Parameters:
            data (DataBase): The DataBase object containing the feature matrix and metadata.
            layer_in (str): The layer from which to get the features.
            layer_out (str): The layer to store the processed features.

        Returns:
            DataBase: Transformed data with highly correlated features dropped.
        """
        if self.dropped_features_ is None:
            raise ValueError("The estimator must be fit before transforming the data.")
        if data is None or not isinstance(data, DataBase):
            raise TypeError("data must be a DataBase object.")
        if layer_in not in ["raw", "processed"]:
            raise ValueError("layer_in must be 'raw' or 'processed'.")
        if layer_out not in ["raw", "processed"]:
            raise ValueError("layer_out must be 'raw' or 'processed'.")
        if data.get_features(layer=layer_in) is None:
            raise ValueError("The input data must contain features.")

        X = data.get_features(layer=layer_in).copy()
        if not all(col in X.columns for col in self.dropped_features_):
            raise ValueError(
                "The selected features must be a subset of the features in the input data."
            )

        X_dropped = X.drop(columns=self.dropped_features_)
        data.update_features(X_dropped, layer=layer_out)
        return data

    def _identify_highly_correlated_features(
        self, X: pd.DataFrame, y: pd.Series
    ) -> list:
        """
        Identify features with high correlation based on the specified correlation threshold.

        Parameters:
            X (pd.DataFrame): The feature matrix with feature names.
            y (pd.Series): The target variable.
        Returns:
            to_drop: List of feature names to be dropped due to high correlation.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X should be of type pd.DataFrame!")
        if not isinstance(y, pd.Series):
            raise TypeError("y should be of type pd.Series!")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        if y.name is None:
            y.name = "label"
        
        print("Identifying correlated features ...", flush=True)

        corr_matrix = X.merge(y, left_index=True, right_index=True).corr().abs()
        corr_X, corr_y = corr_matrix.iloc[:-1, :-1], corr_matrix.iloc[:-1, -1]
        to_drop = []
        for col in X.columns:
            corr = corr_X[col][corr_X[col] >= self.feature_corr_threshold]
            if len(corr) > 1:
                # Keep feature with the highest correlation with y
                keep = corr_y[corr.index].idxmax()
                to_drop.extend(corr.index.drop(keep))

        return list(set(to_drop))

