###############################################################
## Copyright: PromptBio Corp 2023
# Author: whgu, jiayu
# Date of creation: 08/08/2024
# Date of revision: 10/11/2024
## AutoML
## Description: The file define the AutoML utils.
#
###############################################################
#!/usr/bin/env python3

from enum import Enum
from typing import List, Dict, Tuple, Union


import pandas as pd
import numpy as np
import scanpy as sc
from scipy.stats import zscore
import scipy.stats as stats
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    median_absolute_error,
    r2_score,
    average_precision_score,
)

from .DataSampler import DataSamplerSplit
from .Data import DataBase


def read_gene_name_mapper():
    """
    Read the gene name mapper to map gene names to different naming system
    """
    pass


class HPOAlgorithm(Enum):
    """
    Definition of HPOAlgorithm.
    """

    GridSearch = "grid_search"
    RandomSearch = "random_search"
    BayesianSearch = "bayesian_search"


class Metrics(Enum):
    """
    Definition of metrics.
    """

    Accuracy = "accuracy"
    Fscore = "f1"
    Precision = "precision"
    Recall = "recall"
    AUC = "roc_auc"
    CorrCoef = "correlation_coefficient"
    MAE = "neg_mean_absolute_error"
    MSE = "neg_mean_squared_error"
    R2 = "r2"
    PR_AUC = "pr_auc"


class Average(Enum):
    """
    Definition of average.
    """

    Binary = "binary"
    Micro = "micro"
    Macro = "macro"
    Weighted = "weighted"


class AnalysisType(Enum):
    """
    Definition of AnalysisType.
    """

    Classification = "classification"
    Regression = "regression"
    Survival = "survival"


def calculate_prediction_performance(
    y_true,
    y_pred,
    metrics: Metrics = Metrics.Accuracy,
    average: Average = Average.Macro,
) -> float:
    """
    Implement accuracy, fscore, precision and recall to measure classification performance.
    These implementations are work in both binary classification and multilabel case.

    Parmas:
        y_true (1d array-like): Ground truth (correct) target values.
        y_pred (1d array-like): Estimated targets as returned by a classifier.
        metrics (Metrics = Metrics.Accuracy): Metrics to measure classification performance.
                If Metrics.Accuracy, then compute accuracy classification score.
                If Metrics.Fscore, then compute the F1 score.
                If Metrics.Precision, then compute the precision.
                If Metrics.Recall, then compute the recall.
                If Metrics.AUC, then compute the ROC AUC.
                if Metrics.PR_AUC, then compute the average precision score.
        average (Average = Average.Micro): When the metrics in [Metrics.Fscore, Metrics.Precision, Metrics.Recall, Metrics.PR_AUC],
                average determines the type of averaging performed on the data.
                If Average.Binary, then only report results for the class == 1.
                This is applicable only if targets (y_{true,pred}) are binary.
                If Average.Micro, then calculate metrics globally by counting the total true positives,
                false negatives and false positives.
                If Average.Macro, then calculate metrics for each label, and find their unweighted mean.
                This does not take label imbalance into account.
                If Average.Weighted, then calculate metrics for each label, and find their average weighted by
                support (the number of true instances for each label).
    Return:
        calculated_metrics (float): Calculated result.
    """
    # Check if input are valid.
    if not isinstance(metrics, Metrics):
        raise TypeError("metrics should be of type Metrics!")
    if not isinstance(average, Average):
        raise TypeError("average should be of type Average!")

    if metrics == Metrics.Accuracy:
        return accuracy_score(y_true, y_pred)
    elif metrics == Metrics.Fscore:
        return f1_score(y_true, y_pred, average=average.value)
    elif metrics == Metrics.Precision:
        return precision_score(y_true, y_pred, average=average.value)
    elif metrics == Metrics.Recall:
        return recall_score(y_true, y_pred, average=average.value)
    elif metrics == Metrics.PR_AUC:
        return average_precision_score(y_true, y_pred, average=average.value)
    elif metrics == Metrics.AUC:
        return roc_auc_score(y_true, y_pred, average=average.value, multi_class="ovr")
    elif metrics == Metrics.CorrCoef:
        return stats.pearsonr(y_true, y_pred)[0]
    elif metrics == Metrics.MAE:
        return median_absolute_error(y_true, y_pred)
    elif metrics == Metrics.MSE:
        return float(np.mean((y_true - y_pred) ** 2))
    elif metrics == Metrics.R2:
        return r2_score(y_true, y_pred)


def get_scoring_str(metrics: Metrics, average: Average) -> str:
    """
    This function is to check if input `metrics` is of type Metrics and `average` is type of Average.
    Then return the scoring parameters for cross validation according to the `metrics` and `average`.

    Params:
        metrics: Metrics to measure model's performance.
        average: When the metrics in [Metrics.Fscore, Metrics.Precision, Metrics.Recall],
                average determines the type of averaging performed on the data.
    Return:
        scoring (str): Define the strategy to evaluate the performance of the cross-validation.
    """
    if metrics and not isinstance(metrics, Metrics):
        raise TypeError("metrics should be of type Metrics!")
    if average and not isinstance(average, Average):
        raise TypeError("average should be of type Average!")

    if metrics is None:
        return None

    if (
        metrics == Metrics.Accuracy
        or metrics == Metrics.AUC
        or metrics == Metrics.MAE
        or metrics == Metrics.MSE
        or metrics == Metrics.R2
        or average == Average.Binary
        or average is None
    ):
        scoring = metrics.value
    else:
        scoring = metrics.value + "_" + average.value

    return scoring


def check_X_y(X, y) -> None:
    """
    This function is to check if input `X` is of type pd.DataFrame and `y` is of type np.ndarray.

    Params:
        X: Input training vectors.
        y: Class labels.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X should be of type pd.DataFrame!")
    if not isinstance(y, np.ndarray) and not isinstance(y, pd.Series):
        raise TypeError("y should be of type np.ndarray or pd.Series!")


def convert_survival_label(labels: pd.DataFrame) -> np.ndarray:
    """
    This function converts the survival label into the format that can be used by the model.

    Params:
        labels (pd.DataFrame): Survival data.

    Return:
        y (np.ndarray of shape (n_samples, Tuple)): Survival data.
    """
    # Check if input are valid.
    if not isinstance(labels, pd.DataFrame):
        raise TypeError("labels should be of type pd.DataFrame!")
    if "event" not in labels.columns:
        raise ValueError("labels should contain column 'event'!")
    if "time" not in labels.columns:
        raise ValueError("labels should contain column 'time'!")

    y = np.array(
        [tuple(item) for item in labels[["event", "time"]].values],
        dtype=[("event", "?"), ("time", "<f8")],
    )

    return y


def calculate_confusion_matrix(y_true, y_pred) -> Tuple[float, float, float, Tuple]:
    """
    Calculate confusion matrix.

    Params:
        y_true (1d array-like): Ground truth (correct) target values.
        y_pred (1d array-like): Estimated targets as returned by a classifier.
    Return:
        accuracy (float): Accuracy of the model.
        sensitivity (float): Sensitivity of the model.
        specificity (float): Specificity of the model.
        confusion_matrix (Tuple): Confusion matrix.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # Get the accuracy.
    accuracy = calculate_prediction_performance(
        y_true=y_true, y_pred=y_pred, metrics=Metrics.Accuracy
    )

    return accuracy, sensitivity, specificity, (int(tn), int(fp), int(fn), int(tp))


def batch_effect_normalization(feature_matrix: pd.DataFrame):
    """
    Perform batch effect normalization.

    Params:
        feature_matrix (pd.DataFrame): Feature matrix of different batches.
    Return:
        normalized_feature_matrix (pd.DataFrame): Normalized feature matrix.
    """
    print("Performing batch effect normalization...")
    # Check the input.
    if not isinstance(feature_matrix, pd.DataFrame):
        raise TypeError("feature_matrix should be of type pd.DataFrame!")
    if "batch" not in feature_matrix.columns:
        raise ValueError("feature_matrix should have a column named 'batch'!")

    # Normalize the feature matrix with ComBat.
    adata = sc.AnnData(
        X=feature_matrix.drop(columns=["batch"]), obs=feature_matrix[["batch"]]
    )
    sc.pp.combat(adata, key="batch")
    normalized_feature_matrix = pd.DataFrame(
        adata.X, columns=feature_matrix.columns[:-1]
    )

    # Check if the normalized feature matrix is all NaN.
    if normalized_feature_matrix.isnull().all().all():
        return feature_matrix.drop(columns=["batch"])
    else:
        return normalized_feature_matrix


def get_statistic_matrix(
    feature_matrix: pd.DataFrame,
    outlier_threshold: Union[float, int] = 3.5,
) -> pd.DataFrame:
    """
    Compute statistic index for input feature matrix

    Params:
        data (pd.DataFrame): Raw data.
        outlier_threshold (Union[float, int] = 3.5): If the absolute value of z-score > outlier_threshold, it is an outlier.
    Return:
        feat_meta_info (pd.DataFrame): Statistic index matrix.
    """
    meta_info = [
        "feature",
        "type",
        "mean",
        "median",
        "25%_percentile",
        "75%_percentile",
        "min",
        "max",
        "standard_deviation",
        "skewness",
        "NA_ratio",
        "outlier_ratio",
    ]
    feat_meta_info = pd.DataFrame(index=feature_matrix.columns, columns=meta_info)
    feat_meta_info["feature"] = feat_meta_info.index

    # Distinguish between continuous features and categorical features.
    unique_counts = feature_matrix.nunique()
    continuous_features = feature_matrix.loc[:, unique_counts > 2].columns
    categorical_features = feature_matrix.loc[:, unique_counts <= 2].columns
    feat_meta_info.loc[continuous_features, "type"] = "Continuous"
    feat_meta_info.loc[categorical_features, "type"] = "Binary"

    # compute statistis index for continuous featuresss
    feat_meta_info["mean"] = feature_matrix[continuous_features].mean()
    feat_meta_info["median"] = feature_matrix[continuous_features].quantile()
    feat_meta_info["25%_percentile"] = feature_matrix[continuous_features].quantile(
        0.25
    )
    feat_meta_info["75%_percentile"] = feature_matrix[continuous_features].quantile(
        0.75
    )
    feat_meta_info["min"] = feature_matrix[continuous_features].min()
    feat_meta_info["max"] = feature_matrix[continuous_features].max()
    feat_meta_info["standard_deviation"] = feature_matrix[continuous_features].std(
        axis=0
    )
    feat_meta_info["skewness"] = feature_matrix[continuous_features].skew(axis=0)
    feat_meta_info["NA_ratio"] = feature_matrix.isna().mean(axis=0)

    # Calculate the z-score of each feature.
    if len(continuous_features) > 0:
        z_scores = np.abs(zscore(feature_matrix.loc[:, continuous_features]))

        # Check how many outliers are found in the feature.
        feat_meta_info["outlier_ratio"] = (z_scores > outlier_threshold).mean(axis=0)
    return feat_meta_info

def prepare_train_test_data(data: DataBase, test_ratio: float, random_state: int)-> List[DataBase]:
    """
    Prepare the train and test data.

    Params:
        data: The multi-omics data.
        test_ratio: The ratio of the test data.
        random_state: The random state.
    Returns:
        train_data: The train data.
        test_data: The test data.
    """
    if test_ratio < 0 or test_ratio > 1:
        raise ValueError("test_ratio must be a float between 0 and 1")
    if not isinstance(random_state, int):
        raise ValueError("random_state must be an integer")
    if not isinstance(data, DataBase):
        raise ValueError("data must be an instance of DataBase")
    
    sampler = DataSamplerSplit(
        test_size=test_ratio, 
        data = data, 
        random_state=random_state
        )
    train_data, test_data = sampler[0]
    return train_data, test_data
