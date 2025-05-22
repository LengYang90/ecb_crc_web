#!/usr/bin/env python3
###############################################################
## Copyright: PromptBio Corp 2024
## Author: Wenhao Gu
## Date of creation: 07/23/2024
## Date of revision: 10/28/2024
#
## Project: MLGenie
## Description: This file defines the base class for analysis.
##
###############################################################
import os
import random
import numpy as np
from abc import abstractmethod
from abc import ABC, abstractmethod
from typing import List, Tuple

from sklearn.base import BaseEstimator,TransformerMixin

from ..Data import MultiOmicsData, BinaryData, ContinousData, DataBase
from ..DataProcessor import (
    BinaryDataProcessor,
    ContinuesDataProcessor,
    DataProcessorBase
)



class AnalysisBase(BaseEstimator, TransformerMixin):
    def __init__(self, 
        result_dir: str = None, 
        random_state: int = 123,
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
        ):
        """
        Initialize the analysis base.

        Params:
            result_dir(str): The directory to save the result.
            random_state(int): Random state for reproducing.
            null_ratio_threshold(float): The threshold for null ratio.
            dominance_threshold(float): The threshold for dominance.
            impute_strategy(str): The strategy for imputation.
            impute_fill_value(float): The fill value for imputation.   
            chi2_p_threshold(float): The threshold for chi2 p-value.
            variance_threshold(float): The threshold for variance.
            fold_change_threshold(float): The threshold for fold change.
            target_corr_threshold(float): The threshold for target correlation.
            skew_threshold(float): The threshold for skew.
            skew_correction_method(str): The method for skew correction.
            outlier_detection_method(str): The method for outlier detection.
            outlier_detection_threshold(float): The threshold for outlier detection.
            outlier_correction_method(str): The method for outlier correction.
            scaling_method(str): The method for scaling.
            scaling_feature_range(Tuple[float, float]): The range for scaling.
            feature_corr_threshold(float): The threshold for feature correlation.
        """
        if result_dir is not None:
            if not isinstance(result_dir, str):
                raise TypeError("result_dir must be a string")
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
        
        # Check random_state
        if not isinstance(random_state, int):
            raise TypeError("random_state must be an integer")
        if random_state < 0:
            raise ValueError("random_state must be a non-negative integer")
        self.result_dir = result_dir
        self.random_state = random_state

        self.null_ratio_threshold = null_ratio_threshold
        self.dominance_threshold = dominance_threshold
        self.impute_strategy = impute_strategy
        self.impute_fill_value = impute_fill_value
        self.chi2_p_threshold = chi2_p_threshold
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

        # Set random state.
        random.seed(random_state)
        os.environ["PYTHONHASHSEED"] = str(random_state)
        np.random.seed(random_state)


    def _get_processor(self, data: DataBase):
        """
        Get the data processor for the data.
        
        Params:
            data: The data.
        Returns:
            processor: The data processor.
        """
        if isinstance(data, BinaryData):
            processor = BinaryDataProcessor(
            null_ratio_threshold=self.null_ratio_threshold,
            dominance_threshold=self.dominance_threshold,
            chi2_p_threshold=self.chi2_p_threshold,
            )
        elif isinstance(data, ContinousData):
            processor = ContinuesDataProcessor(
                null_ratio_threshold=self.null_ratio_threshold,
                variance_threshold=self.variance_threshold,
                fold_change_threshold=self.fold_change_threshold,
                target_corr_threshold=self.target_corr_threshold,
                skew_threshold=self.skew_threshold,
                skew_correction_method=self.skew_correction_method,
                outlier_detection_method=self.outlier_detection_method,
                outlier_detection_threshold=self.outlier_detection_threshold,
                outlier_correction_method=self.outlier_correction_method,
                scaling_method=self.scaling_method,
                scaling_feature_range=self.scaling_feature_range,
                feature_corr_threshold=self.feature_corr_threshold,
        )
        else:
            processor = DataProcessorBase(
                    null_ratio_threshold=self.null_ratio_threshold,
                )
        return processor

    def _preprocess(self, data: DataBase, is_train: bool):
        """"
        Preprocess the multi-omics data.

        Params:
            data: The multi-omics data.
            is_train: The Whether it is training data.
        Returns:
            processed_data: The preprocessed multi-omics data.
        """
        if data.if_processed:
            return data
        if is_train:
            self.processor = {}
            if isinstance(data, MultiOmicsData):
                for omics_type, omics_data in data.multi_omics_data.items():
                    processor = self._get_processor(omics_data)
                    processor.fit(omics_data)
                    omics_data = omics_data.update_processed_flag(inplace=True)
                    self.processor[omics_type] = processor
                data.update_processed_flag(inplace=True)
            else:
                processor = self._get_processor(data)
                processor.fit(data)
                data = data.update_processed_flag(inplace=True)
                self.processor = processor
        else:
            if isinstance(data, MultiOmicsData):
                for omics_type, omics_data in data.multi_omics_data.items():
                    processor = self.processor[omics_type]
                    omics_data = processor.transform(omics_data)
                    omics_data = omics_data.update_processed_flag(inplace=True)
                data.update_processed_flag(inplace=True)
            else:
                data = self.processor.transform(data)
                data = data.update_processed_flag(inplace=True)
        return data


    # @abstractmethod
    def fit(self, data: MultiOmicsData):
        raise NotImplementedError("The fit method must be implemented by subclasses.")

    # @abstractmethod
    def transform(self, data: MultiOmicsData):
        raise NotImplementedError("The transform method must be implemented by subclasses.")

    def fit_transform(self, data: DataBase):
        self.fit(data)
        return self.transform(data)
    
    # @abstractmethod
    def save_result(self):
        raise NotImplementedError("The save_result method must be implemented by subclasses.")
