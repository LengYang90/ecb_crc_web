###############################################################
## Copyright: PromptBio Corp 2022
# Author: whgu
# Date of creation: 11/29/2022
# Date of revision: 12/06/2023
#
## AIM
## Description: Visualization of AIM.
###############################################################
from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve



def plot_performance(
    classifier: BaseEstimator,
    test_X: pd.DataFrame,
    test_y: np.ndarray,
    roc: bool = True,
    pos_label: int = 1,
):
    """
    Plot Receiver operating characteristic (ROC) curve and Precision Recall Curve.
    For binary classification:
        a. ROC curve.
        b. P-R curve.
    For multi class classification:
        a. ROC curve with given label of the positive class.
        b. P-R curve with given label of the positive class.

    Params:
        classifier (BaseEstimator): Trained classifier.
        test_X (pd.DataFrame of shape (n_samlpes, n_features)): Features of test set.
        test_y (np.ndarray of shape (n_samlpes, 1)): Labels of test set.
        roc (bool = True):
            True: Plot Receiver operating characteristic (ROC) curve.
            False: Plot Precision Recall Curve.
        pos_label (int = 1): The class considered as the positive class.
        average (Average = Average.Binary): average determines the type of averaging performed on the data.
                If Average.Binary, then only report results for the class == pos_label.
                If Average.Micro, then calculate metrics globally by counting the total true positives,
                false negatives and false positives.
                If Average.Macro, then calculate metrics for each label, and find their mean.
    Returns:
        fpr or precision (list of float): False positive rate or Precision.
        tpr or recall (list of float): True positive rate or Recall.
        thresholds (list of float): Thresholds used to compute fpr-tpr or precision-recall.
    """
    # Check if input are valid.
    if not isinstance(classifier, BaseEstimator):
        raise TypeError("Classifier must be of type BaseEstimator!")
    if not isinstance(test_X, pd.DataFrame):
        raise TypeError("test_X should be of type pd.DataFrame!")
    if not isinstance(test_y, np.ndarray):
        raise TypeError("test_y should be of type np.ndarray!")
    if pos_label not in test_y:
        raise ValueError("Positive label doesn't exist in test_y!")

    predict_y = classifier.predict_proba(test_X)
    pos_class_idx = classifier.classes_.tolist().index(pos_label)
    if roc:
        fpr, tpr, thresholds = roc_curve(
            y_true=test_y, y_score=predict_y[:, pos_class_idx], pos_label=pos_label
        )
        fpr = [round(i, 2) for i in fpr]
        tpr = [round(i, 2) for i in tpr]
        thresholds = thresholds.tolist()
        return fpr, tpr, thresholds

    else:
        precision, recall, thresholds = precision_recall_curve(
            y_true=test_y, probas_pred=predict_y[:, pos_class_idx], pos_label=pos_label
        )
        precision = [round(i, 2) for i in precision]
        recall = [round(i, 2) for i in recall]
        thresholds = thresholds.tolist()
        return precision, recall, thresholds



def get_performance_fig(
    classifier: BaseEstimator, test_X: np.ndarray, test_y: np.ndarray,
):
    """
    Plot and encode model's performance.

    Params:
        classifier (BaseEstimator): Trained classifier.
        test_X (np.ndarray of shape (n_samlpes, n_features)): Features of test set.
        test_y (np.ndarray of shape (n_samlpes, 1)): Labels of test set.
    Returns:
        ROC_data (tuple): (fpr, tpr, thresholds) of ROC curve.
        PR_data (tuple): (precision, recall, thresholds) of P-R curve.
    """
    # Plot ROC curve
    ROC_data = plot_performance(classifier, test_X, test_y, pos_label=1)
    # Plot P-R curve
    PR_data = plot_performance(classifier, test_X, test_y, pos_label=1, roc=False)
    return ROC_data, PR_data

