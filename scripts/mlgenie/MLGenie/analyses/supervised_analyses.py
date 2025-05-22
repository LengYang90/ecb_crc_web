#!/usr/bin/env python3
###############################################################
## Copyright: PromptBio Corp 2024
## Author: Wenhao Gu, Jiayu Chen
## Date of creation: 07/23/2024
## Date of revision: 12/23/2024
## Project: MLGenie
## Description: This file defines the class for classification analysis.
##
###############################################################
import os
import sys
from typing import List, Union, Tuple
from abc import abstractmethod

import shap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ._base import AnalysisBase
from ..model.classification import ClsModelType
from ..model.model_utils import compare_models
from ..Data import MultiOmicsData,DataBase
from ..FeatureSelector import FeatureSelector
from ..utils import Metrics, HPOAlgorithm, AnalysisType, calculate_confusion_matrix, batch_effect_normalization
from ..DataSampler import DataSamplerSplit
from ..visualization import get_performance_fig
from ..Utils.Utils import compute_feature_shap_values

class SupervisedAnalysis(AnalysisBase):
    def __init__(
        self,
        result_dir: str = None,
        random_state: int = 123,
        null_ratio_threshold: float=0.5,
        dominance_threshold: float=0.95,
        chi2_p_threshold: float=0.5,
        variance_threshold: float=0.01,
        fold_change_threshold: float=None,
        target_corr_threshold: float=0,
        skew_threshold: float=None,
        skew_correction_method: str="yeo-johnson",
        outlier_detection_method: str="z_score",
        outlier_detection_threshold: float=3,
        outlier_correction_method: str="clip",
        scaling_method: str="standard",
        scaling_feature_range: Tuple[float, float]=(0, 1),
        feature_corr_threshold: float=0,
        metrics: Metrics = Metrics.AUC,
        n_feature_to_select: int = 10,
        specified_models: List[str] = None,
        hpo_algorithm: HPOAlgorithm = HPOAlgorithm.GridSearch,
        hpo_search_iter: int = 100,
        cv: int = 5,
        n_jobs: int = 8,
        n_bootstrap: int = 50,
    ):
        """
        Initialize the supervised analysis base.

        Params:
            metrics: The metrics for evaluation.
            n_feature_to_select: The number of features to select.
            specified_models: The specified models to compare.
            hpo_algorithm: The hyperparameter optimization algorithm.
            hpo_search_iter: The number of iterations for hyperparameter optimization.
            cv: The number of cross-validation folds.
            n_jobs: The number of jobs to run in parallel.
            n_bootstrap: The number of bootstrap iterations.
        """
        if n_feature_to_select is None:
            n_feature_to_select = 10
        self._check_parameters(
            metrics = metrics,
            n_feature_to_select = n_feature_to_select,
            specified_models = specified_models,
            hpo_algorithm = hpo_algorithm,
            hpo_search_iter = hpo_search_iter,
            cv = cv,
            n_jobs = n_jobs,
            n_bootstrap = n_bootstrap
        )

        self.metrics = metrics
        self.n_feature_to_select = n_feature_to_select
        self.specified_models = specified_models
        self.hpo_algorithm = hpo_algorithm
        self.hpo_search_iter = hpo_search_iter
        self.cv = cv
        self.n_jobs = n_jobs
        self.n_bootstrap = n_bootstrap

        super().__init__(
            result_dir = result_dir,
            random_state = random_state,
            null_ratio_threshold = null_ratio_threshold,
            dominance_threshold = dominance_threshold,
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
            feature_corr_threshold = feature_corr_threshold,
            )
    def _check_parameters(
        self,
        metrics: Metrics,
        n_feature_to_select: int,
        specified_models: List[str],
        hpo_algorithm: HPOAlgorithm,
        hpo_search_iter: int,
        cv: int,
        n_jobs: int,
        n_bootstrap
    ):
        """
        Check the parameters for the supervised analysis.
        """

        # Check the metrics.
        if not isinstance(metrics, Metrics):
            raise TypeError("metrics must be an instance of Metrics")

        # Check the n_feature_to_select.
        if not isinstance(n_feature_to_select, int) or n_feature_to_select <= 0:
            raise ValueError("n_feature_to_select must be None or a positive integer")

        # Check specified_models
        if specified_models is not None:
            if not isinstance(specified_models, list):
                raise TypeError("specified_models must be a list or None")
            if not all(isinstance(model, str) for model in specified_models):
                raise ValueError("All elements in specified_models must be strings")

        # Check hpo_algorithm
        if not isinstance(hpo_algorithm, HPOAlgorithm):
            raise TypeError("hpo_algorithm must be an instance of HPOAlgorithm")

        # Check hpo_search_iter
        if not isinstance(hpo_search_iter, int) or hpo_search_iter <= 0:
            raise ValueError("hpo_search_iter must be a positive integer")

        # Check cv
        if not isinstance(cv, int) or cv <= 1:
            raise ValueError("cv must be an integer greater than 1")

        # Check n_jobs
        if n_jobs is not None:
            if not isinstance(n_jobs, int):
                raise TypeError("n_jobs must be an integer or None")
            if n_jobs < -1 or n_jobs == 0:
                raise ValueError("n_jobs must be a positive integer, -1, or None")

        # Check n_bootstrap
        if not isinstance(n_bootstrap, int) or n_bootstrap <= 0:
            raise ValueError("n_bootstrap must be a positive integer")

    def _batch_effect_normalization(self, data: DataBase):
        """
        Batch effect normalization.

        Params:
            data: The multi-omics data.
        Returns:
            normalized_data: The normalized multi-omics data.
        """
        feature_matrix = data.get_features("raw")
        if "batch" in data.get_feature_names():
            normalized_feature_matrix = batch_effect_normalization(feature_matrix)
        else:
            normalized_feature_matrix = feature_matrix
        data.update_features(normalized_feature_matrix, layer="processed")
        return data

    def _feature_selection(self, data: DataBase)-> List[str]:
        """
        Select features for classification analysis.

        Params:
            data: The multi-omics data.
        Returns:
            selected_features: The selected features.
        """
        X = data.get_features("processed")
        y = data.get_labels()
        feature_selector = FeatureSelector(
                task_type=self.task_type,
                n_features =self.n_feature_to_select if self.n_feature_to_select else "auto",
                random_state=self.random_state,
                n_jobs = self.n_jobs,
                n_bootstrap = self.n_bootstrap
                )
        feature_selector.fit(X, y)
        selected_features, selected_scores = feature_selector.get_selected_features()

        if self.result_dir:
            feature_selector.feature_scores_.to_csv(os.path.join(self.result_dir, "feature_scores.csv"))
        return selected_features, selected_scores

    def _model_selection(self, data: DataBase) -> ClsModelType:
        """
        Select model for classification analysis.

        Params:
            X: The feature matrix.
            y: The label.
        Returns:
            model: The selected model.
        """
        model_comparison = compare_models(
            train_X = data.get_features("processed"),
            train_y = data.get_labels(),
            analysis_type = self.analysis_type,
            specified_models = self.specified_models,
            metrics = self.metrics,
            random_state = self.random_state,
            cv = self.cv,
            n_jobs = self.n_jobs
        )
        model_comparison.sort(key=lambda t: t[1])
        selected_model, best_metric = model_comparison[-1]
        model_comparison = [(item[0].name, item[1]) for item in model_comparison]
        return model_comparison, selected_model, best_metric

    def _compute_feature_shap_values(self, selected_feature_matrix: pd.DataFrame):
        """
        Compute the SHAP values for the features.
        """
        shap_dir = os.path.join(self.result_dir, 'shap')
        if not os.path.exists(shap_dir):
            os.makedirs(shap_dir)

        shap_obj, shap_data = compute_feature_shap_values(self.model.model, selected_feature_matrix, self.task_type)

        #  use shap to plot SHAP dot plot for all features
        plt.figure()
        shap.summary_plot(
            shap_values=shap_obj,
            features=selected_feature_matrix,
            plot_type="dot",
            show=False,
            feature_names=[
                feature.split("|")[-1] for feature in selected_feature_matrix.columns
            ],
        )
        plt.savefig(os.path.join(shap_dir, 'shap_summary_plot.png'), bbox_inches='tight')
        plt.close()

        # plot SHAP plot using SHAP values
        stripplot = sns.stripplot(data=shap_data, x="shap_value", y="feature", hue="feature_scale")
        offsets = stripplot.collections[0].get_offsets()
        offsets_df = pd.DataFrame(offsets, columns=["x", "y"])
        offsets_df.to_csv(os.path.join(shap_dir, 'shap_plot_data.csv'), index=False)

        # use shap to plot SHAP dependence plots
        for column in selected_feature_matrix.columns:
            column_name = column.replace("|","_")
            plt.figure()
            shap.plots.scatter(shap_obj[:, column], show=False)
            plt.xlabel(column.split("|")[-1])
            plt.ylabel("SHAP value")
            plt.savefig(os.path.join(shap_dir, f'shap_scatter_plot_{column_name}.png'), bbox_inches='tight')
            plt.close()

        # save the shap data
        shap_data.to_csv(os.path.join(shap_dir, 'shap_data.csv'), index=False)


class ClsAnalysis(SupervisedAnalysis):
    def __init__(
        self,
        result_dir: str = None,
        random_state: int = 123,
        null_ratio_threshold: float=0.5,
        dominance_threshold: float=0.95,
        chi2_p_threshold: float=0.5,
        variance_threshold: float=0.01,
        fold_change_threshold: float=None,
        target_corr_threshold: float=0,
        skew_threshold: float=None,
        skew_correction_method: str="yeo-johnson",
        outlier_detection_method: str="z_score",
        outlier_detection_threshold: float=3,
        outlier_correction_method: str="clip",
        scaling_method: str="standard",
        scaling_feature_range: Tuple[float, float]=(0, 1),
        feature_corr_threshold: float=0,
        metrics: Metrics = Metrics.AUC,
        n_feature_to_select: int = 10,
        specified_models: List[str] = None,
        hpo_algorithm: HPOAlgorithm = HPOAlgorithm.GridSearch,
        hpo_search_iter: int = 100,
        cv: int = 5,
        n_jobs: int = 8,
        n_bootstrap: int = 50,
        control_group: Union[str, int] = None,
    ):
        """
        Initialize the classification analysis.

        Params:
            control_group: The control group.
        """
        # Check the control_group
        if control_group is not None:
            if not isinstance(control_group, (str, int)):
                raise TypeError("control_group must be a string or integer")

        self.control_group = control_group
        self.task_type = "classification"
        self.analysis_type = AnalysisType.Classification

        self.model_comparison = None
        self.best_metric = None

        super().__init__(
            result_dir = result_dir,
            random_state = random_state,
            null_ratio_threshold = null_ratio_threshold,
            dominance_threshold = dominance_threshold,
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
            feature_corr_threshold = feature_corr_threshold,
            metrics = metrics,
            n_feature_to_select = n_feature_to_select,
            specified_models = specified_models,
            hpo_algorithm = hpo_algorithm,
            hpo_search_iter = hpo_search_iter,
            cv = cv,
            n_jobs = n_jobs,
            n_bootstrap=n_bootstrap
        )

    def _check_control_group(self, data: DataBase):
        """
        Check the control group.

        Params:
            data: The data.
        """
        labels = data.get_labels()
        if self.control_group and self.control_group not in labels.unique():
           raise ValueError(f"control_group {self.control_group} not found in labels")

        if self.control_group is None:
            if set(labels.unique()) != {0, 1}:
                raise ValueError("When control_group is None, labels must be {0,1}.")

    def _convert_label_to_binary(self, data: DataBase) -> np.ndarray:
        """
        This function is to convert label to binary label.

        Params:
            labels (pd.Series): Label column.
            control_group (str or int): Control group.
        """
        labels = data.get_labels()
        if self.control_group is not None:
            converted_labels = labels.apply(lambda t: 0 if t == self.control_group else 1)
            data.update_labels(converted_labels, inplace= True)
        return data

    def fit(self, data: DataBase):
        """
        Fit the classification analysis to the data.

        Params:
            data: The multi-omics data.
        """
        if not isinstance(data, DataBase):
            raise ValueError("data must be an instance of DataBase")

        # Check the control group.
        self._check_control_group(data)
        print("--------_check_control_group done!----------------\n\n")

        # Convert the label to binary
        data = self._convert_label_to_binary(data)
        print("--------_convert_label_to_binary done!----------------\n\n")

        # Preprocess the data.
        data = self._preprocess(data, is_train=True)
        print("--------_preprocess done!----------------\n\n")

        # Feature selection
        self.selected_features, self.selected_scores = self._feature_selection(data)
        print("--------_feature_selection done!----------------\n\n")

        # Update the features
        selected_feature_matrix = data.get_features("processed", selected_features = self.selected_features)
        selected_data = DataBase(
            features = selected_feature_matrix,
            labels = data.get_labels(),
            if_processed = True
        )
        # Model selection
        self.model_comparison, selected_model, best_metric = self._model_selection(selected_data)
        print("--------_model_selection done!----------------\n\n")

        params = {
            "cv": self.cv,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            "metrics": self.metrics,
        }
        self.model = selected_model.value
        self.model.set_params(**params)

        # Train the model
        self.model.fit(
            X=selected_feature_matrix,
            y=data.get_labels()
            )
        print("--------_fit done!----------------\n\n")

        # Visualize the SHAP values
        # selected_raw_feature_matrix = data.get_raw_features(selected_features = self.selected_features)
        self._compute_feature_shap_values(selected_feature_matrix)
        print("----------------------save feature_shap_values done!--------------")

        return self

    def transform(self, data: DataBase):
        # Check if the model has been fitted
        if not hasattr(self, 'model'):
            raise ValueError("Model has not been fitted yet. Please call fit() before transform().")

        if not isinstance(data, DataBase):
            raise ValueError("data must be an instance of DataBase")

        # Check the control group.
        self._check_control_group(data)
        print("--------_check_control_group done!----------------\n\n")

        # Convert the label to binary
        data = self._convert_label_to_binary(data)
        print("--------_convert_label_to_binary done!----------------\n\n")

        # Preprocess the data.
        data = self._preprocess(data, is_train=False)
        print("--------_preprocess done!----------------\n\n")

        # Update the features
        selected_feature_matrix = data.get_features("processed", selected_features = self.selected_features)

        X = selected_feature_matrix
        y = data.get_labels()

        # Predict the label
        pred_label = self.model.predict(X)

        # Calculate the performance
        (
            accuracy,
            sensitivity,
            specificity,
            confusion_matrix,
        ) = calculate_confusion_matrix(y, pred_label)

        performance = {
            "auc": self.model.evaluate(X, y, metrics=Metrics.AUC),
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "confusion_matrix": confusion_matrix,
            "pr_auc": self.model.evaluate(X, y, metrics=Metrics.PR_AUC),
            "precision": self.model.evaluate(X, y, metrics=Metrics.Precision),
            "recall": self.model.evaluate(X, y, metrics=Metrics.Recall),
            "f1_score": self.model.evaluate(X, y, metrics=Metrics.Fscore),
        }
        # Visualize the best model's performance.
        ROC_data, PR_data = get_performance_fig(self.model.model, X, y.to_numpy())


        return (
            pd.DataFrame(self.model.predict_proba(X)[:,1], columns=["prediction"], index=X.index),
            performance,
            ROC_data,
            PR_data
        )

    def save_result(self):
        pass
