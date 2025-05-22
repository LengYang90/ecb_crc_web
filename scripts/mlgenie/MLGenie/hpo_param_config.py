###############################################################
## Copyright: PromptBio Corp 2022
# Author: whgu, jiayu
# Date of creation: 04/23/2022
# Date of revision: 08/08/2024
## AIM
## Description: Definition of ParamConfig;
#
###############################################################
from enum import Enum
from typing import List, Dict, Tuple, Union

import scipy.stats as stats
from skopt.space import Real, Categorical, Integer

from .utils import HPOAlgorithm


class ClsParamConfig:
    """
    Define the hyper parameter configuration space.
    """

    @staticmethod
    def get_param_config(
        model_type: str, hpo_algotithm: HPOAlgorithm = HPOAlgorithm.GridSearch
    ) -> List[Dict]:
        """
        Get param_config according to the ModelType and HPOAlgorithm.

        Params:
            model_type (str): specified model type
            hpo_algotithm (HPOAlgorithm = HPOAlgorithm.GridSearch) : GridSearch, RandomizedSearch or BayesianSearch.
        Returns:
            param_config (List[Dict[str]]): A list of dictionary with parameter
            names as keys and lists of parameter settings to try as values.
        """
        # Check if input are valid.
        if not isinstance(model_type, str):
            raise TypeError("model_type should be of type string!")
        if not isinstance(hpo_algotithm, HPOAlgorithm):
            raise TypeError("hpo_algotithm should be of type HPOAlgorithm!")

        if model_type == "SVM":
            return ClsParamConfig.SVM_param_config(hpo_algotithm)
        elif model_type == "LR":
            return ClsParamConfig.LR_param_config(hpo_algotithm)
        elif model_type == "KNN":
            return ClsParamConfig.KNN_param_config(hpo_algotithm)
        elif model_type == "RF":
            return ClsParamConfig.RF_param_config(hpo_algotithm)
        elif model_type == "DT":
            return ClsParamConfig.DT_param_config(hpo_algotithm)
        elif model_type == "GBDT":
            return ClsParamConfig.GBDT_param_config(hpo_algotithm)
        elif model_type == "MLP":
            return ClsParamConfig.MLP_param_config(hpo_algotithm)
        else:
            raise ValueError("Unsupported model_type {}".format(model_type))

    @staticmethod
    def SVM_param_config(hpo_algotithm: HPOAlgorithm) -> List[Dict]:
        """
        Define the hyper parameter configuration space of SVM.

        Hyper parameters:
        kernel: SVM kernel type {"Linear Kernel", "Polynomial Kernel",
            "Gaussian Kernel", "Sigmoid Kernel"},
        C: float, Default = 1.0
           Penalty parameter C of the error term.
           The larger C is, the worse the generalization ability is, and the phenomenon of over-fitting is prone to occur;
           The smaller the C is, the better the generalization ability is, and the phenomenon of under-fitting is prone to occur.
        gamma: {"scale", "auto"} or float, default = "scale"
            Kernel coefficient for "rbf", "poly" and "sigmoid".
            If "scale", then use 1 / (n_features * X.var()) as value of gamma.
            If "auto", then use 1 / n_features.
            The larger γ is, the better the training set fits, the worse the generalization ability,
            and the phenomenon of over-fitting is prone to occur.
        tol: float, Tolerance for stopping criterion.
        Params:
            hpo_algotithm (HPOAlgorithm=HPOAlgorithm.GridSearch) : GridSearch, RandomizedSearch or BayesianSearch
                If GridSearch, then return param list;
                If RandomizedSearch, then return param distributions;
                If BayesianSearch, then return param range.
        Returns:
            param_config (List[Dict[str, List]]) :
                A list of dictionary with parameter names as keys and lists of parameter settings to try as values.
        """
        # gamma is kernel coefficient for "rbf", "poly" and "sigmoid",
        # which becomes irrelevant when kernel == "linear".
        if hpo_algotithm == HPOAlgorithm.GridSearch:
            param_config = [
                {
                    "kernel": ["linear"],
                    "C": [0.01, 0.1, 1, 10, 100, 1000],
                    "gamma": ["scale"],
                    "tol": [1e-3, 1e-4, 1e-5],
                },
                {
                    "kernel": ["rbf", "sigmoid", "poly"],
                    "C": [0.01, 0.1, 1, 10, 100, 1000],
                    "gamma": ["auto", "scale", 1, 0.1, 0.01, 0.001],
                    "tol": [1e-3, 1e-4, 1e-5],
                },
            ]
        elif hpo_algotithm == HPOAlgorithm.RandomSearch:
            param_config = [
                {
                    "kernel": ["linear"],
                    "C": stats.uniform(0.01, 100),
                    "gamma": ["scale"],
                    "tol": stats.uniform(1e-3, 1e-5),
                },
                {
                    "kernel": ["rbf", "sigmoid", "poly"],
                    "C": stats.uniform(0.01, 100),
                    "gamma": ["auto", "scale", 1, 0.1, 0.01, 0.001],
                    "tol": stats.uniform(1e-3, 1e-5),
                },
            ]
        elif hpo_algotithm == HPOAlgorithm.BayesianSearch:
            param_config = [
                {
                    "kernel": Categorical(["linear"]),
                    "C": Real(0.01, 100),
                    "gamma": Categorical(["scale"]),
                    "tol": Real(1e-5, 1e-3),
                },
                {
                    "kernel": Categorical(["rbf", "sigmoid", "poly"]),
                    "C": Real(0.01, 100),
                    "gamma": Real(0.001, 1),
                    "tol": Real(1e-5, 1e-3),
                },
            ]

        return param_config

    @staticmethod
    def LR_param_config(hpo_algotithm: HPOAlgorithm) -> List[Dict]:
        """
        Define the hyper parameter configuration space of LogisticRegression.

        Hyper parameters:
        penalty: {"l1", "l2", "elasticnet", "none"}, default = "l2"
            Specify the norm of the penalty.
            'none': no penalty is added;
            'l2': L2 penalty term and it is the default choice;
            'l1': L1 penalty term;
            'elasticnet': using both L1 and L2 penalty terms. Can only be used with saga solver
        tol: float, Tolerance for stopping criterion.
        C: float, default = 1.0
            Inverse of regularization strength.
            Smaller values specify stronger regularization.
        multi_class: {"auto", "ovr", "multinomial"}, default = "auto"
            If the option chosen is "ovr", then a binary problem is fit for each label.
            For "multinomial" the loss minimised is the multinomial loss fit across the entire probability distribution,
            even when the data is binary.
        solver: optimization algorithm -- {"newton-cg", "lbfgs", "liblinear",
            "sag", "saga"}, default = "lbfgs"
            For small datasets, 'liblinear' is a good choice, whereas 'sag' and 'saga' are faster for large ones;
            For multiclass problems, only 'newton-cg', 'sag', 'saga' and 'lbfgs' handle multinomial loss;
        max_iter: int, default = 100
            Maximum number of iterations taken for the solvers to converge.
        l1_ratio: float,
            Only used if penalty='elasticnet'.
            Setting l1_ratio=0 is equivalent to using penalty='l2', while setting l1_ratio=1 is equivalent to using penalty='l1'.
            For 0 < l1_ratio <1, the penalty is a combination of L1 and L2
        Params:
            hpo_algotithm (HPOAlgorithm=HPOAlgorithm.GridSearch) : GridSearch, RandomizedSearch or BayesianSearch
                If GridSearch, then return param list;
                If RandomizedSearch, then return param distributions;
                If BayesianSearch, then return param range.
        Returns:
            param_config (List[Dict[str]]):
                A list of dictionary with parameters names as keys and lists of
                parameter settings to try as values.
        """
        # The choice of the algorithm depends on the penalty chosen:
        # "newton-cg" - ["l2", "none"]
        # "lbfgs" - ["l2", "none"]
        # "liblinear" - ["l1", "l2"]
        # "sag" - ["l2", "none"]
        # "saga" - ["elasticnet", "l1", "l2", "none"]
        # "liblinear" is limited to one-versus-rest schemes.
        if hpo_algotithm == HPOAlgorithm.GridSearch:
            param_config = [
                {
                    "penalty": ["l2"],
                    "tol": [1e-3, 1e-4, 1e-5],
                    "C": [0.01, 0.1, 1, 10, 100],
                    "multi_class": ["ovr", "multinomial"],
                    "solver": ["newton-cg", "lbfgs", "sag", "saga"],
                    "max_iter": [100, 200, 500, 1000],
                    "l1_ratio": [None],
                },
                {
                    "penalty": ["l1"],
                    "tol": [1e-3, 1e-4, 1e-5],
                    "C": [0.01, 0.1, 1, 10, 100],
                    "multi_class": ["ovr"],
                    "solver": ["liblinear", "saga"],
                    "max_iter": [100, 200, 500, 1000],
                    "l1_ratio": [None],
                },
                {
                    "penalty": ["elasticnet"],
                    "tol": [1e-3, 1e-4, 1e-5],
                    "C": [0.01, 0.1, 1, 10, 100],
                    "multi_class": ["ovr", "multinomial"],
                    "solver": ["saga"],
                    "max_iter": [100, 200, 500, 1000],
                    "l1_ratio": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                },
            ]
        elif hpo_algotithm == HPOAlgorithm.RandomSearch:
            param_config = [
                {
                    "penalty": ["l2"],
                    "tol": stats.uniform(1e-3, 1e-5),
                    "C": stats.uniform(0.01, 100),
                    "multi_class": ["ovr", "multinomial"],
                    "solver": ["newton-cg", "lbfgs", "sag", "saga"],
                    "max_iter": range(100, 1000, 100),
                    "l1_ratio": [None],
                },
                {
                    "penalty": ["l1"],
                    "tol": stats.uniform(1e-3, 1e-5),
                    "C": stats.uniform(0.01, 100),
                    "multi_class": ["ovr"],
                    "solver": ["liblinear", "saga"],
                    "max_iter": range(100, 1000, 100),
                    "l1_ratio": [None],
                },
                {
                    "penalty": ["elasticnet"],
                    "tol": stats.uniform(1e-3, 1e-5),
                    "C": stats.uniform(0.01, 100),
                    "multi_class": ["ovr", "multinomial"],
                    "solver": ["saga"],
                    "max_iter": range(100, 1000, 100),
                    "l1_ratio": stats.uniform(0, 1),
                },
            ]
        elif hpo_algotithm == HPOAlgorithm.BayesianSearch:
            param_config = [
                {
                    "penalty": Categorical(["l2"]),
                    "tol": Real(1e-5, 1e-3),
                    "C": Real(0.01, 100),
                    "multi_class": Categorical(["ovr", "multinomial"]),
                    "solver": Categorical(["newton-cg", "lbfgs", "sag", "saga"]),
                    "max_iter": Integer(100, 1000),
                    "l1_ratio": [None],
                },
                {
                    "penalty": Categorical(["l1"]),
                    "tol": Real(1e-5, 1e-3),
                    "C": Real(0.01, 100),
                    "multi_class": Categorical(["ovr"]),
                    "solver": Categorical(["liblinear", "saga"]),
                    "max_iter": Integer(100, 1000),
                },
                {
                    "penalty": Categorical(["elasticnet"]),
                    "tol": Real(1e-5, 1e-3),
                    "C": Real(0.01, 100),
                    "multi_class": Categorical(["ovr", "multinomial"]),
                    "solver": Categorical(["saga"]),
                    "max_iter": Integer(100, 1000),
                    "l1_ratio": Real(0, 1),
                },
            ]

        return param_config

    @staticmethod
    def RF_param_config(hpo_algotithm: HPOAlgorithm) -> List[Dict]:
        """
        Define the hyper parameter configuration space of Random Forest.

        Hyper parameters:
        n_estimators: int, Default = 100
            The number of trees in the forest.
            The larger n_estimators is, the phenomenon of over-fitting is prone to occur;
            The smaller the n_estimators is, the phenomenon of under-fitting is prone to occur.
        bootstrap: bool, Default = True
            Whether bootstrap samples are used when building trees.
        criterion: {"gini", "entropy"}, default = "gini"
            The function to measure the quality of a split.
        max_depth: int, default = None
            Maximum depth of decision tree.
            The value can be set to 10-100 with large sample size or a large number of features.
        max_features: {"auto", "sqrt", "log2"}, default = "auto"
            The number of features to consider when looking for the best split.
            If None, then max_features = n_features.
        min_samples_leaf: int or float, default = 1
            The minimum number of samples required to be at a leaf node.
            If the number of leaf nodes is less than the number of samples, they will be pruned together with sibling nodes.

        Params:
            hpo_algotithm (HPOAlgorithm=HPOAlgorithm.GridSearch) : GridSearch, RandomizedSearch or BayesianSearch
                If GridSearch, then return param list;
                If RandomizedSearch, then return param distributions;
                If BayesianSearch, then return param range.
        Returns:
            param_config (List[Dict[str]]): A list of dictionary with parameters names as keys and lists
                                            of parameter settings to try as values.
        """
        if hpo_algotithm == HPOAlgorithm.GridSearch:
            param_config = [
                {
                    "n_estimators": [10, 50, 100, 150, 200, 500],
                    "bootstrap": [True, False],
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 3, 4, 5, 6, 7, 8, 9, 10, 20],
                    "max_features": [None, "sqrt", "log2"],
                    "min_samples_leaf": [1, 2, 4, 8, 10, 20],
                }
            ]
        elif hpo_algotithm == HPOAlgorithm.RandomSearch:
            param_config = [
                {
                    "n_estimators": stats.randint(10, 1000),
                    "bootstrap": [True, False],
                    "criterion": ["gini", "entropy"],
                    "max_depth": stats.randint(1, 100),
                    "max_features": [None, "sqrt", "log2"],
                    "min_samples_leaf": stats.randint(1, 20),
                }
            ]
        elif hpo_algotithm == HPOAlgorithm.BayesianSearch:
            param_config = [
                {
                    "n_estimators": Integer(10, 1000),
                    "bootstrap": Categorical([True, False]),
                    "criterion": Categorical(["gini", "entropy"]),
                    "max_depth": Integer(1, 100),
                    "max_features": Categorical([None, "sqrt", "log2"]),
                    "min_samples_leaf": Integer(1, 20),
                }
            ]

        return param_config

    @staticmethod
    def KNN_param_config(hpo_algotithm: HPOAlgorithm) -> List[Dict]:
        """
        Define the hyper parameter configuration space of KNN.

        Hyper parameters:
        n_neighbors: int, default = 5
            Number of neighbors to use by default for kneighbors queries.
        weights: {"uniform", "distance"}, default='uniform'
            Weight function used in prediction.
            "uniform" : uniform weights. All points in each neighborhood are weighted equally.
            "distance" : weight points by the inverse of their distance.
            In this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.

        Params:
            hpo_algotithm (HPOAlgorithm=HPOAlgorithm.GridSearch) : GridSearch, RandomizedSearch or BayesianSearch
                If GridSearch, then return param list;
                If RandomizedSearch, then return param distributions;
                If BayesianSearch, then return param range.
        Returns:
            param_config (List[Dict[str]]): A list of dictionary with parameters names as keys and lists of parameter settings to try as values.
        """
        if hpo_algotithm == HPOAlgorithm.GridSearch:
            param_config = [
                {
                    "n_neighbors": [i for i in range(1, 51)],
                    "weights": ["uniform", "distance"],
                }
            ]
        elif hpo_algotithm == HPOAlgorithm.RandomSearch:
            param_config = [
                {"n_neighbors": range(1, 51), "weights": ["uniform", "distance"]}
            ]
        elif hpo_algotithm == HPOAlgorithm.BayesianSearch:
            param_config = [
                {
                    "n_neighbors": Integer(1, 50),
                    "weights": Categorical(["uniform", "distance"]),
                }
            ]

        return param_config

    @staticmethod
    def DT_param_config(hpo_algotithm: HPOAlgorithm) -> List[Dict]:
        """
        Define the hyper parameter configuration space of Decision Tree.

        Hyper parameters:
        criterion: {"gini", "entropy"}, default = "gini"
            The function to measure the quality of a split.
            "gini" for the Gini impurity and “entropy” for the information gain.
        splitter: {"best", "random"}, default="best"
            The strategy used to choose the split at each node.
            Supported strategies are "best" to choose the best split and "random" to choose the best random split.
        max_depth: int, default = None
            Maximum depth of decision tree.
            The value can be set to 10-100 with large sample size or a large number of features.
        max_features: {"auto", "sqrt", "log2"}, default = None
            The number of features to consider when looking for the best split.
            If None, then max_features = n_features.
        min_samples_leaf: int or float, default = 1
            The minimum number of samples required to be at a leaf node.
            If the number of leaf nodes is less than the number of samples, they will be pruned together with sibling nodes.
        min_samples_split: int, default=2
            The minimum number of samples required to split an internal node.
        Params:
            hpo_algotithm (HPOAlgorithm=HPOAlgorithm.GridSearch) : GridSearch, RandomizedSearch or BayesianSearch
                If GridSearch, then return param list;
                If RandomizedSearch, then return param distributions;
                If BayesianSearch, then return param range.
        Returns:
            param_config (List[Dict[str]]): A list of dictionary with parameters names as keys and lists of parameter settings to try as values.
        """
        if hpo_algotithm == HPOAlgorithm.GridSearch:
            param_config = [
                {
                    "criterion": ["gini", "entropy"],
                    "splitter": ["best", "random"],
                    "max_depth": [None, 3, 4, 5, 6, 7, 8, 9, 10, 20],
                    "max_features": [None, "sqrt", "log2"],
                    "min_samples_leaf": [1, 2, 4, 8, 10, 20],
                    "min_samples_split": [2, 5, 10, 20],
                }
            ]
        elif hpo_algotithm == HPOAlgorithm.RandomSearch:
            param_config = [
                {
                    "criterion": ["gini", "entropy"],
                    "splitter": ["best", "random"],
                    "max_depth": stats.randint(1, 50),
                    "max_features": [None, "sqrt", "log2"],
                    "min_samples_leaf": stats.randint(1, 20),
                    "min_samples_split": stats.randint(2, 20),
                }
            ]
        elif hpo_algotithm == HPOAlgorithm.BayesianSearch:
            param_config = [
                {
                    "criterion": Categorical(["gini", "entropy"]),
                    "splitter": Categorical(["best", "random"]),
                    "max_depth": Integer(1, 50),
                    "max_features": Categorical([None, "sqrt", "log2"]),
                    "min_samples_leaf": Integer(1, 20),
                    "min_samples_split": Integer(2, 20),
                }
            ]
        return param_config

    @staticmethod
    def GBDT_param_config(hpo_algotithm: HPOAlgorithm) -> List[Dict]:
        """
        Define the hyper parameter configuration space of Gradient Boosting Classifier.

        Hyper parameters:
        n_estimators: int, default=100,
            The number of boosting stages to perform.
            Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.
        learning_rate: float, default=0.1
            Learning rate shrinks the contribution of each tree by learning_rate.
            There is a trade-off between learning_rate and n_estimators.
        loss: {"deviance"}, default = "deviance"
            The loss function to be optimized.
            "deviance" refers to deviance (= logistic regression) for classification with probabilistic outputs.
        max_depth: int, default = None
            Maximum depth of decision tree.
            The value can be set to 10-100 with large sample size or a large number of features.
        max_features: {"auto", "sqrt", "log2"}, default = None
            The number of features to consider when looking for the best split.
            If None, then max_features = n_features.
        min_samples_leaf: int or float, default = 1
            The minimum number of samples required to be at a leaf node.
            If the number of leaf nodes is less than the number of samples, they will be pruned together with sibling nodes.
        min_samples_split: int
            The minimum number of samples required to split an internal node.
        Params:
            hpo_algotithm (HPOAlgorithm=HPOAlgorithm.GridSearch) : GridSearch, RandomizedSearch or BayesianSearch
                If GridSearch, then return param list;
                If RandomizedSearch, then return param distributions;
                If BayesianSearch, then return param range.
        Returns:
            param_config (List[Dict[str]]): A list of dictionary with parameters names as keys and lists
                                            of parameter settings to try as values.
        """
        if hpo_algotithm == HPOAlgorithm.GridSearch:
            param_config = [
                {
                    "n_estimators": [10, 50, 100, 150, 200, 500],
                    "learning_rate": [0.0001, 0.001, 0.01, 0.1],
                    "loss": ["log_loss", "exponential"],
                    "max_depth": [None, 3, 4, 5, 6, 7, 8, 9, 10, 20],
                    "max_features": [None, "sqrt", "log2"],
                    "min_samples_leaf": [1, 2, 4, 8, 10, 20],
                    "min_samples_split": [2, 5, 10, 20],
                }
            ]
        elif hpo_algotithm == HPOAlgorithm.RandomSearch:
            param_config = [
                {
                    "n_estimators": stats.randint(10, 1000),
                    "learning_rate": stats.uniform(0.0001, 0.1),
                    "loss": ["log_loss", "exponential"],
                    "max_depth": stats.randint(1, 100),
                    "max_features": [None, "sqrt", "log2"],
                    "min_samples_leaf": stats.randint(1, 20),
                    "min_samples_split": stats.randint(2, 20),
                }
            ]
        elif hpo_algotithm == HPOAlgorithm.BayesianSearch:
            param_config = [
                {
                    "n_estimators": Integer(10, 1000),
                    "learning_rate": Real(0.0001, 0.1),
                    "loss": Categorical(["log_loss", "exponential"]),
                    "max_depth": Integer(1, 100),
                    "max_features": Categorical([None, "sqrt", "log2"]),
                    "min_samples_leaf": Integer(1, 20),
                    "min_samples_split": Integer(2, 20),
                }
            ]

        return param_config

    @staticmethod
    def MLP_param_config(hpo_algotithm: HPOAlgorithm) -> List[Dict]:
        """
        Define the hyper parameter configuration space of multi-layer perceptron Classifier.

        Hyper parameters:
        hidden_layer_sizes: tuple, length = n_layers - 2, default=(100,)
            The ith element represents the number of neurons in the ith hidden layer.
        activation: {"logistic", "tanh", "relu"}, default="relu"
            Activation function for the hidden layer.
        alpha: float, default=0.0001
            Strength of the L2 regularization term.
        batch_size: int,
            Size of minibatches for stochastic optimizers.
            When set to "auto", batch_size=min(200, n_samples).
        learning_rate_init: float, default=0.001
            The initial learning rate used. It controls the step-size in updating the weights.
        max_iter: int, default = 200
            Number of epochs.

        Params:
            hpo_algotithm (HPOAlgorithm) :  GridSearch or RandomizedSearch
                If GridSearchCV, then return param list;
                If RandomizedSearchCV, then return param distributions.
        Return:
            param_config (List[Dict[str]]): A list of dictionary with parameters names as keys and lists
                                            of parameter settings to try as values.
        """
        if hpo_algotithm == HPOAlgorithm.GridSearch:
            param_config = [
                {
                    "hidden_layer_sizes": [
                        (50,),
                        (100,),
                        (200,),
                        (100, 50),
                        (200, 100),
                        (200, 100, 50),
                    ],
                    "activation": ["tanh", "relu", "logistic"],
                    "alpha": [0.00001, 0.0001, 0.001],
                    "batch_size": [16, 32, 64],
                    "learning_rate_init": [0.0001, 0.0005, 0.001, 0.005, 0.01],
                    "max_iter": [100, 200],
                }
            ]
        elif hpo_algotithm == HPOAlgorithm.RandomSearch:
            param_config = [
                {
                    "hidden_layer_sizes": [
                        (50,),
                        (100,),
                        (200,),
                        (100, 50),
                        (200, 100),
                        (200, 100, 50),
                    ],
                    "activation": ["tanh", "relu", "logistic"],
                    "alpha": stats.uniform(0.00001, 0.001),
                    "batch_size": [16, 32, 64],
                    "learning_rate_init": stats.uniform(0.0001, 0.01),
                    "max_iter": [100, 200],
                }
            ]
        elif hpo_algotithm == HPOAlgorithm.BayesianSearch:
            param_config = [
                {
                    "hidden_layer_sizes": list(range(50, 201, 10)),
                    "activation": Categorical(["tanh", "relu", "logistic"]),
                    "alpha": Real(0.00001, 0.001),
                    "batch_size": Categorical([16, 32, 64]),
                    "learning_rate_init": Real(0.0001, 0.01),
                    "max_iter": Categorical([100, 200]),
                }
            ]

        return param_config


class RegressionParamConfig:
    """
    Define the hyper parameter configuration space.
    """

    @staticmethod
    def get_param_config(
        model_type: str,
        hpo_algotithm: HPOAlgorithm = HPOAlgorithm.GridSearch,
    ) -> List[Dict]:
        """
        Get param_config according to the RegressionModelType and HPOAlgorithm.

        Params:
            model_type (str): specified model type.
            hpo_algotithm (HPOAlgorithm = HPOAlgorithm.GridSearch) : GridSearch, RandomizedSearch or BayesianSearch.
        Returns:
            param_config (List[Dict[str]]): A list of dictionary with parameter
            names as keys and lists of parameter settings to try as values.
        """
        # Check if input are valid.
        if not isinstance(model_type, str):
            raise TypeError("model_type should be of type string!")
        if not isinstance(hpo_algotithm, HPOAlgorithm):
            raise TypeError("hpo_algotithm should be of type HPOAlgorithm!")

        if model_type == "LassoR":
            return RegressionParamConfig.Lasso_param_config(hpo_algotithm)
        elif model_type == "RidgeR":
            return RegressionParamConfig.Ridge_param_config(hpo_algotithm)
        elif model_type == "ElasticR":
            return RegressionParamConfig.ElasticNet_param_config(hpo_algotithm)
        elif model_type == "GBDTR":
            return RegressionParamConfig.GBDT_param_config(hpo_algotithm)
        elif model_type == "RFR":
            return RegressionParamConfig.RF_param_config(hpo_algotithm)
        elif model_type == "DTR":
            return RegressionParamConfig.DT_param_config(hpo_algotithm)
        else:
            raise ValueError("Unsupported model_type {}".format(model_type))

    @staticmethod
    def Lasso_param_config(hpo_algotithm: HPOAlgorithm) -> List[Dict]:
        """
        Define the hyper parameter configuration space of SVM.

        Hyper parameters:
        alpha (float): Weights of penalty.
        max_iter (int): Maximum number of iterations for solvers to converge.
        tol (float): The tolerance for the optimization.
        warm_start (bool): When set to True, reuse the solution of the previous call to fit as initialization.
        selection (str): If set to ‘random’, a random coefficient is updated every iteration rather than looping over features sequentially by default.

        Params:
            hpo_algotithm (HPOAlgorithm=HPOAlgorithm.GridSearch) : GridSearch, RandomizedSearch or BayesianSearch
                If GridSearch, then return param list;
                If RandomizedSearch, then return param distributions;
                If BayesianSearch, then return param range.
        Returns:
            param_config (List[Dict[str, List]]) :
                A list of dictionary with parameter names as keys and lists of parameter settings to try as values.
        """
        if hpo_algotithm == HPOAlgorithm.GridSearch:
            param_config = [
                {
                    "alpha": [0.001, 0.01, 0.1, 1, 10, 100],
                    "max_iter": [100, 200, 500, 1000],
                    "tol": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                    "warm_start": [True, False],
                    "selection": ["cyclic", "random"],
                }
            ]
        elif hpo_algotithm == HPOAlgorithm.RandomSearch:
            param_config = [
                {
                    "alpha": stats.uniform(0.001, 100),
                    "max_iter": range(100, 1000, 100),
                    "tol": stats.uniform(1e-6, 1e-2),
                    "warm_start": [True, False],
                    "selection": ["cyclic", "random"],
                }
            ]
        elif hpo_algotithm == HPOAlgorithm.BayesianSearch:
            param_config = [
                {
                    "alpha": Real(0.001, 100),
                    "max_iter": list(range(100, 1000, 100)),
                    "tol": Real(1e-6, 1e-2),
                    "warm_start": Categorical([True, False]),
                    "selection": Categorical(["cyclic", "random"]),

                }
            ]

        return param_config

    @staticmethod
    def ElasticNet_param_config(hpo_algotithm: HPOAlgorithm) -> List[Dict]:
        """
        Define the hyper parameter configuration space of LogisticRegression.

        Hyper parameters:
        alpha (float): Weight of penalty.
        l1_ratio (float): The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1.
            For l1_ratio = 0 the penalty is an L2 penalty.
            For l1_ratio = 1 it is an L1 penalty.
            For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
        max_iter (int): Maximum number of iterations for solvers to converge.
        tol (float): The tolerance for the optimization.
        warm_start (bool): When set to True, reuse the solution of the previous call to fit as initialization.
        selection (str): If set to ‘random’, a random coefficient is updated every iteration rather than looping over features sequentially by default.

        Params:
            hpo_algotithm (HPOAlgorithm=HPOAlgorithm.GridSearch) : GridSearch, RandomizedSearch or BayesianSearch
                If GridSearch, then return param list;
                If RandomizedSearch, then return param distributions;
                If BayesianSearch, then return param range.
        Returns:
            param_config (List[Dict[str]]):
                A list of dictionary with parameters names as keys and lists of
                parameter settings to try as values.
        """
        if hpo_algotithm == HPOAlgorithm.GridSearch:
            param_config = [
                {
                    "alpha": [0.001, 0.01, 0.1, 1, 10, 100],
                    "l1_ratio": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                    "max_iter": [100, 200, 500, 1000],
                    "tol": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                    "warm_start": [True, False],
                    "selection": ["cyclic", "random"],
                }
            ]
        elif hpo_algotithm == HPOAlgorithm.RandomSearch:
            param_config = [
                {
                    "alpha": stats.uniform(0.001, 100),
                    "l1_ratio": stats.uniform(0, 1),
                    "max_iter": range(100, 1000, 100),
                    "tol": stats.uniform(1e-6, 1e-2),
                    "warm_start": [True, False],
                    "selection": ["cyclic", "random"],
                }
            ]
        elif hpo_algotithm == HPOAlgorithm.BayesianSearch:
            param_config = [
                {
                    "alpha": Real(0.001, 100),
                    "l1_ratio": Real(0, 1),
                    "max_iter": list(range(100, 1000, 100)),
                    "tol": Real(1e-6, 1e-2),
                    "warm_start": Categorical([True, False]),
                    "selection": Categorical(["cyclic", "random"]),
                }
            ]

        return param_config

    @staticmethod
    def Ridge_param_config(hpo_algotithm: HPOAlgorithm) -> List[Dict]:
        """
        Define the hyper parameter configuration space of LogisticRegression.

        Hyper parameters:
        alpha (float): Constant that multiplies the L2 term, controlling regularization strength.
        tol (float): The tolerance for the optimization.
        solver (str): Solver to use in the computational routines.
        Params:
            hpo_algotithm (HPOAlgorithm=HPOAlgorithm.GridSearch) : GridSearch, RandomizedSearch or BayesianSearch
                If GridSearch, then return param list;
                If RandomizedSearch, then return param distributions;
                If BayesianSearch, then return param range.
        Returns:
            param_config (List[Dict[str]]):
                A list of dictionary with parameters names as keys and lists of
                parameter settings to try as values.
        """
        if hpo_algotithm == HPOAlgorithm.GridSearch:
            param_config = [
                {
                    "alpha": [0, 0.001, 0.01, 0.1, 1, 10, 100],
                    "tol": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                    "solver": [
                        "auto",
                        "svd",
                        "cholesky",
                        "lsqr",
                        "sparse_cg",
                        "sag",
                        "saga",
                        "lbfgs",
                    ],
                }
            ]
        elif hpo_algotithm == HPOAlgorithm.RandomSearch:
            param_config = [
                {
                    "alpha": stats.uniform(0, 100),
                    "tol": stats.uniform(1e-6, 1e-2),
                    "solver": Categorical(
                        [
                            "auto",
                            "svd",
                            "cholesky",
                            "lsqr",
                            "sparse_cg",
                            "sag",
                            "saga",
                            "lbfgs",
                        ]
                    ),
                }
            ]
        elif hpo_algotithm == HPOAlgorithm.BayesianSearch:
            param_config = [
                {
                    "alpha": Real(0, 100),
                    "tol": Real(1e-6, 1e-2),
                    "solver": Categorical(
                        [
                            "auto",
                            "svd",
                            "cholesky",
                            "lsqr",
                            "sparse_cg",
                            "sag",
                            "saga",
                            # "lbfgs",
                        ]
                    ),
                }
            ]

        return param_config

    @staticmethod
    def GBDT_param_config(hpo_algotithm: HPOAlgorithm) -> List[Dict]:
        """
        Define the hyper parameter configuration space of Gradient Boosting Classifier.

        Hyper parameters:
        n_estimators: int, default=100,
            The number of boosting stages to perform.
            Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.
        learning_rate: float, default=0.1
            Learning rate shrinks the contribution of each tree by learning_rate.
            There is a trade-off between learning_rate and n_estimators.
        loss: {"deviance"}, default = "deviance"
            The loss function to be optimized.
            "deviance" refers to deviance (= logistic regression) for classification with probabilistic outputs.
        max_depth: int, default = None
            Maximum depth of decision tree.
            The value can be set to 10-100 with large sample size or a large number of features.
        max_features: {"auto", "sqrt", "log2"}, default = None
            The number of features to consider when looking for the best split.
            If None, then max_features = n_features.
        min_samples_leaf: int or float, default = 1
            The minimum number of samples required to be at a leaf node.
            If the number of leaf nodes is less than the number of samples, they will be pruned together with sibling nodes.
        min_samples_split: int
            The minimum number of samples required to split an internal node.
        Params:
            hpo_algotithm (HPOAlgorithm=HPOAlgorithm.GridSearch) : GridSearch, RandomizedSearch or BayesianSearch
                If GridSearch, then return param list;
                If RandomizedSearch, then return param distributions;
                If BayesianSearch, then return param range.
        Returns:
            param_config (List[Dict[str]]): A list of dictionary with parameters names as keys and lists
                                            of parameter settings to try as values.
        """
        if hpo_algotithm == HPOAlgorithm.GridSearch:
            param_config = [
                {
                    "n_estimators": [10, 50, 100, 150, 200, 500],
                    "learning_rate": [0.0001, 0.001, 0.01, 0.1],
                    "loss": ["squared_error", "absolute_error", "huber"],
                    "max_depth": [None, 3, 4, 5, 6, 7, 8, 9, 10, 20],
                    "max_features": [None, "sqrt", "log2"],
                    "min_samples_leaf": [1, 2, 4, 8, 10, 20],
                    "min_samples_split": [2, 5, 10, 20],
                }
            ]
        elif hpo_algotithm == HPOAlgorithm.RandomSearch:
            param_config = [
                {
                    "n_estimators": stats.randint(10, 1000),
                    "learning_rate": stats.uniform(0.0001, 0.1),
                    "loss": ["squared_error", "absolute_error", "huber"],
                    "max_depth": stats.randint(1, 100),
                    "max_features": [None, "sqrt", "log2"],
                    "min_samples_leaf": stats.randint(1, 20),
                    "min_samples_split": stats.randint(2, 20),
                }
            ]
        elif hpo_algotithm == HPOAlgorithm.BayesianSearch:
            param_config = [
                {
                    "n_estimators": Integer(10, 1000),
                    "learning_rate": Real(0.0001, 0.1),
                    "loss": Categorical(["squared_error", "absolute_error", "huber"]),
                    "max_depth": Integer(1, 100),
                    "max_features": Categorical([None, "sqrt", "log2"]),
                    "min_samples_leaf": Integer(1, 20),
                    "min_samples_split": Integer(2, 20),
                }
            ]

        return param_config

    @staticmethod
    def RF_param_config(hpo_algotithm: HPOAlgorithm) -> List[Dict]:
        """
        Define the hyper parameter configuration space of Random Forest.

        Hyper parameters:
        n_estimators: int, Default = 100
            The number of trees in the forest.
            The larger n_estimators is, the phenomenon of over-fitting is prone to occur;
            The smaller the n_estimators is, the phenomenon of under-fitting is prone to occur.
        bootstrap: bool, Default = True
            Whether bootstrap samples are used when building trees.
        criterion: {"gini", "entropy"}, default = "gini"
            The function to measure the quality of a split.
        max_depth: int, default = None
            Maximum depth of decision tree.
            The value can be set to 10-100 with large sample size or a large number of features.
        max_features: {"auto", "sqrt", "log2"}, default = "auto"
            The number of features to consider when looking for the best split.
            If None, then max_features = n_features.
        min_samples_leaf: int or float, default = 1
            The minimum number of samples required to be at a leaf node.
            If the number of leaf nodes is less than the number of samples, they will be pruned together with sibling nodes.

        Params:
            hpo_algotithm (HPOAlgorithm=HPOAlgorithm.GridSearch) : GridSearch, RandomizedSearch or BayesianSearch
                If GridSearch, then return param list;
                If RandomizedSearch, then return param distributions;
                If BayesianSearch, then return param range.
        Returns:
            param_config (List[Dict[str]]): A list of dictionary with parameters names as keys and lists
                                            of parameter settings to try as values.
        """
        if hpo_algotithm == HPOAlgorithm.GridSearch:
            param_config = [
                {
                    "n_estimators": [10, 50, 100, 150, 200, 500],
                    "bootstrap": [True, False],
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                    "max_depth": [None, 3, 4, 5, 6, 7, 8, 9, 10, 20],
                    "max_features": [None, "sqrt", "log2"],
                    "min_samples_leaf": [1, 2, 4, 8, 10, 20],
                }
            ]
        elif hpo_algotithm == HPOAlgorithm.RandomSearch:
            param_config = [
                {
                    "n_estimators": stats.randint(10, 1000),
                    "bootstrap": [True, False],
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                    "max_depth": stats.randint(1, 100),
                    "max_features": [None, "sqrt", "log2"],
                    "min_samples_leaf": stats.randint(1, 20),
                }
            ]
        elif hpo_algotithm == HPOAlgorithm.BayesianSearch:
            param_config = [
                {
                    "n_estimators": Integer(10, 1000),
                    "bootstrap": Categorical([True, False]),
                    "criterion": Categorical(
                        ["squared_error", "friedman_mse", "absolute_error", "poisson"]
                    ),
                    "max_depth": Integer(1, 100),
                    "max_features": Categorical([None, "sqrt", "log2"]),
                    "min_samples_leaf": Integer(1, 20),
                }
            ]

        return param_config

    @staticmethod
    def DT_param_config(hpo_algotithm: HPOAlgorithm) -> List[Dict]:
        """
        Define the hyper parameter configuration space of Decision Tree.

        Hyper parameters:
        criterion: {"gini", "entropy"}, default = "gini"
            The function to measure the quality of a split.
            "gini" for the Gini impurity and “entropy” for the information gain.
        splitter: {"best", "random"}, default="best"
            The strategy used to choose the split at each node.
            Supported strategies are "best" to choose the best split and "random" to choose the best random split.
        max_depth: int, default = None
            Maximum depth of decision tree.
            The value can be set to 10-100 with large sample size or a large number of features.
        max_features: {"auto", "sqrt", "log2"}, default = None
            The number of features to consider when looking for the best split.
            If None, then max_features = n_features.
        min_samples_leaf: int or float, default = 1
            The minimum number of samples required to be at a leaf node.
            If the number of leaf nodes is less than the number of samples, they will be pruned together with sibling nodes.
        min_samples_split: int, default=2
            The minimum number of samples required to split an internal node.
        Params:
            hpo_algotithm (HPOAlgorithm=HPOAlgorithm.GridSearch) : GridSearch, RandomizedSearch or BayesianSearch
                If GridSearch, then return param list;
                If RandomizedSearch, then return param distributions;
                If BayesianSearch, then return param range.
        Returns:
            param_config (List[Dict[str]]): A list of dictionary with parameters names as keys and lists of parameter settings to try as values.
        """
        if hpo_algotithm == HPOAlgorithm.GridSearch:
            param_config = [
                {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                    "splitter": ["best", "random"],
                    "max_depth": [None, 3, 4, 5, 6, 7, 8, 9, 10, 20],
                    "max_features": [None, "sqrt", "log2"],
                    "min_samples_leaf": [1, 2, 4, 8, 10, 20],
                    "min_samples_split": [2, 5, 10, 20],
                }
            ]
        elif hpo_algotithm == HPOAlgorithm.RandomSearch:
            param_config = [
                {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                    "splitter": ["best", "random"],
                    "max_depth": stats.randint(1, 50),
                    "max_features": [None, "sqrt", "log2"],
                    "min_samples_leaf": stats.randint(1, 20),
                    "min_samples_split": stats.randint(2, 20),
                }
            ]
        elif hpo_algotithm == HPOAlgorithm.BayesianSearch:
            param_config = [
                {
                    "criterion": Categorical(
                        ["squared_error", "friedman_mse", "absolute_error", "poisson"]
                    ),
                    "splitter": Categorical(["best", "random"]),
                    "max_depth": Integer(1, 50),
                    "max_features": Categorical([None, "sqrt", "log2"]),
                    "min_samples_leaf": Integer(1, 20),
                    "min_samples_split": Integer(2, 20),
                }
            ]
        return param_config


class SurvivalParamConfig:
    """
    Define the hyperparameter configuration space.
    """

    @staticmethod
    def get_param_config(
        model_type: str,
        hpo_algotithm: HPOAlgorithm = HPOAlgorithm.GridSearch,
    ) -> List[Dict]:
        """
        Get param_config according to the ModelType and HPOAlgorithm.

        Params:
            model_type (str): specified model type
            hpo_algotithm (HPOAlgorithm = HPOAlgorithm.GridSearch) : GridSearch, RandomizedSearch or BayesianSearch.
        Returns:
            param_config (List[Dict[str]]): A list of dictionary with parameter
            names as keys and lists of parameter settings to try as values.
        """
        # Check if input are valid.
        if not isinstance(model_type, str):
            raise TypeError("model_type should be of type SurvivalModelType!")
        if not isinstance(hpo_algotithm, HPOAlgorithm):
            raise TypeError("hpo_algotithm should be of type HPOAlgorithm!")

        if model_type == "CoxPH":
            return SurvivalParamConfig.CoxPH_param_config(hpo_algotithm)
        elif model_type == "Coxnet":
            return SurvivalParamConfig.Coxnet_param_config(hpo_algotithm)
        elif model_type == "RSF":
            return SurvivalParamConfig.RSF_param_config(hpo_algotithm)
        elif model_type == "GBS":
            return SurvivalParamConfig.GBS_param_config(hpo_algotithm)
        elif model_type == "FSVM":
            return SurvivalParamConfig.FSVM_param_config(hpo_algotithm)

    @staticmethod
    def CoxPH_param_config(hpo_algotithm: HPOAlgorithm) -> List[Dict]:
        """
        Define the hyperparameter configuration space of CoxPH.

        Hyperparameters:
        alpha (float): Regularization parameter for ridge regression penalty.
        tol (float): The convergence tolerance.
        n_iter (int): Maximum number of iterations.
        Params:
            hpo_algotithm (HPOAlgorithm=HPOAlgorithm.GridSearch) : GridSearch, RandomizedSearch or BayesianSearch
                If GridSearch, then return param list;
                If RandomizedSearch, then return param distributions;
                If BayesianSearch, then return param range.
        Returns:
            param_config (List[Dict[str, List]]) :
                A list of dictionary with parameter names as keys and lists of parameter settings to try as values.
        """
        if hpo_algotithm == HPOAlgorithm.GridSearch:
            param_config = [
                {
                    "alpha": [0, 0.001, 0.01, 0.1, 1, 10, 100],
                    "tol": [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
                    "n_iter": [50, 100, 500, 1000],
                }
            ]
        elif hpo_algotithm == HPOAlgorithm.RandomSearch:
            param_config = [
                {
                    "alpha": stats.uniform(0, 100),
                    "tol": [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
                    "n_iter": range(50, 1000, 10),
                }
            ]
        elif hpo_algotithm == HPOAlgorithm.BayesianSearch:
            param_config = [
                {
                    "alpha": Real(0, 100),
                    "tol": [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
                    "n_iter": Integer(50, 1000),
                }
            ]

        return param_config

    @staticmethod
    def Coxnet_param_config(hpo_algotithm: HPOAlgorithm) -> List[Dict]:
        """
        Define the hyperparameter configuration space of Coxnet.

        Hyperparameters:
        l1_ratio (float): The ElasticNet mixing parameter, with 0 < l1_ratio <= 1.
            For l1_ratio = 0 the penalty is an L2 penalty;
            For l1_ratio = 1 it is an L1 penalty;
            For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
        tol (float): The convergence tolerance.
        max_iter (int): Maximum number of iterations.
        Params:
            hpo_algotithm (HPOAlgorithm=HPOAlgorithm.GridSearch) : GridSearch, RandomizedSearch or BayesianSearch
                If GridSearch, then return param list;
                If RandomizedSearch, then return param distributions;
                If BayesianSearch, then return param range.
        Returns:
            param_config (List[Dict[str, List]]) :
                A list of dictionary with parameter names as keys and lists of parameter settings to try as values.
        """
        if hpo_algotithm == HPOAlgorithm.GridSearch:
            param_config = [
                {
                    "l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                    "tol": [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
                    "max_iter": [1000, 10000, 100000],
                }
            ]
        elif hpo_algotithm == HPOAlgorithm.RandomSearch:
            param_config = [
                {
                    "l1_ratio": stats.uniform(0.1, 1),
                    "tol": [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
                    "max_iter": range(1000, 100000, 1000),
                }
            ]
        elif hpo_algotithm == HPOAlgorithm.BayesianSearch:
            param_config = [
                {
                    "l1_ratio": Real(0.1, 1),
                    "tol": [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
                    "max_iter": Integer(1000, 100000),
                }
            ]

        return param_config

    @staticmethod
    def RSF_param_config(hpo_algotithm: HPOAlgorithm) -> List[Dict]:
        """
        Define the hyperparameter configuration space of Random Forest.

        Hyperparameters:
        n_estimators (int): The number of trees in the forest.
        bootstrap (bool): Whether bootstrap samples are used when building trees.
            If False, the whole datset is used to build each tree.
        max_depth (int): The maximum depth of the tree.
        max_features: {"auto", "sqrt", "log2"}, default = "auto"
            The number of features to consider when looking for the best split.
            If None, then max_features = n_features.
        min_samples_leaf: int or float, default = 1
            The minimum number of samples required to be at a leaf node.
            If the number of leaf nodes is less than the number of samples, they will be pruned together with sibling nodes.

        Params:
            hpo_algotithm (HPOAlgorithm=HPOAlgorithm.GridSearch) : GridSearch, RandomizedSearch or BayesianSearch
                If GridSearch, then return param list;
                If RandomizedSearch, then return param distributions;
                If BayesianSearch, then return param range.
        Returns:
            param_config (List[Dict[str, List]]) :
                A list of dictionary with parameter names as keys and lists of parameter settings to try as values.
        """
        if hpo_algotithm == HPOAlgorithm.GridSearch:
            param_config = [
                {
                    "n_estimators": [10, 50, 100, 150, 200, 500],
                    "bootstrap": [True, False],
                    "max_depth": [None, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100],
                    "max_features": [None, "sqrt", "log2"],
                    "min_samples_leaf": [1, 2, 4, 8, 10, 20],
                }
            ]
        elif hpo_algotithm == HPOAlgorithm.RandomSearch:
            param_config = [
                {
                    "n_estimators": stats.randint(10, 1000),
                    "bootstrap": [True, False],
                    "max_depth": stats.randint(1, 100),
                    "max_features": [None, "sqrt", "log2"],
                    "min_samples_leaf": stats.randint(1, 20),
                }
            ]
        elif hpo_algotithm == HPOAlgorithm.BayesianSearch:
            param_config = [
                {
                    "n_estimators": Integer(10, 1000),
                    "bootstrap": Categorical([True, False]),
                    "max_depth": Integer(1, 100),
                    "max_features": Categorical([None, "sqrt", "log2"]),
                    "min_samples_leaf": Integer(1, 20),
                }
            ]

        return param_config

    @staticmethod
    def GBS_param_config(hpo_algotithm: HPOAlgorithm) -> List[Dict]:
        """
        Define the hyperparameter configuration space of Gradient Boosting.

        Hyperparameters:
        loss (str): The loss function to be optimized.
        n_estimators (int): The number of boosting stages to perform.
        learning_rate (float): Learning rate shrinks the contribution of each tree by learning_rate.
        min_samples_split (int or float): The minimum number of samples required to split an internal node.
        min_samples_leaf (int or float): The minimum number of samples required to be at a leaf node.
        max_depth (int): The maximum depth of the tree.
        max_features {"auto", "sqrt", "log2"}: The number of features to consider when looking for the best split.
        criterion (str): The function to measure the quality of a split.
        dropout_rate (float): The dropout rate of the dropout layer.

        Params:
            hpo_algotithm (HPOAlgorithm=HPOAlgorithm.GridSearch) : GridSearch, RandomizedSearch or BayesianSearch
                If GridSearch, then return param list;
                If RandomizedSearch, then return param distributions;
                If BayesianSearch, then return param range.
        Returns:
            param_config (List[Dict[str, List]]) :
                A list of dictionary with parameter names as keys and lists of parameter settings to try as values.
        """
        if hpo_algotithm == HPOAlgorithm.GridSearch:
            param_config = [
                {
                    "loss": ["coxph", "squared", "ipcwls"],
                    "learning_rate": [0.0001, 0.001, 0.01, 0.1],
                    "n_estimators": [10, 50, 100, 200],
                    "criterion": ["friedman_mse"],
                    "max_depth": [None, 3, 5, 7, 9],
                    "max_features": [None, "sqrt", "log2"],
                    "min_samples_leaf": [1, 2, 4, 8],
                    "min_samples_split": [2, 5, 10],
                    "dropout_rate": [0, 0.2, 0.4],
                }
            ]
        elif hpo_algotithm == HPOAlgorithm.RandomSearch:
            param_config = [
                {
                    "loss": ["coxph", "squared", "ipcwls"],
                    "learning_rate": stats.uniform(0.0001, 0.1),
                    "n_estimators": stats.randint(10, 1000),
                    "criterion": ["friedman_mse", "squared_error"],
                    "max_depth": stats.randint(1, 100),
                    "max_features": [None, "sqrt", "log2"],
                    "min_samples_leaf": stats.randint(1, 20),
                    "min_samples_split": stats.randint(2, 20),
                    "dropout_rate": stats.uniform(0, 0.8),
                }
            ]
        elif hpo_algotithm == HPOAlgorithm.BayesianSearch:
            param_config = [
                {
                    # "loss": Categorical(["coxph", "squared", "ipcwls"]),
                    "loss": Categorical(["coxph", "squared"]),  # , "ipcwls"
                    "learning_rate": Real(0.0001, 0.1),
                    "n_estimators": Integer(10, 1000),
                    "criterion": Categorical(["friedman_mse", "squared_error"]),
                    "max_depth": Integer(1, 100),
                    "max_features": Categorical([None, "sqrt", "log2"]),
                    "min_samples_leaf": Integer(1, 20),
                    "min_samples_split": Integer(2, 20),
                    "dropout_rate": Real(0, 0.8),
                }
            ]
        return param_config

    @staticmethod
    def FSVM_param_config(hpo_algotithm: HPOAlgorithm) -> List[Dict]:
        """
        Define the hyperparameter configuration space of Support Vector Machine.

        Hyperparameters:
        alpha (float): Weight of penalizing the squared hinge loss in the objective function.
        rank_ratio (float, optional): Mixing parameter between regression and ranking objective with 0 <= rank_ratio <= 1.
            If rank_ratio = 1, only ranking is performed,
            if rank_ratio = 0, only regression is performed.
        max_iter (int): Maximum number of iterations to perform in Newton optimization.
        optimizer({'avltree', 'direct-count', 'PRSVM', 'rbtree', 'simple'}): Which optimizer to use.
        Params:
            hpo_algotithm (HPOAlgorithm=HPOAlgorithm.GridSearch) : GridSearch, RandomizedSearch or BayesianSearch
                If GridSearch, then return param list;
                If RandomizedSearch, then return param distributions;
                If BayesianSearch, then return param range.
        Returns:
            param_config (List[Dict[str, List]]) :
                A list of dictionary with parameter names as keys and lists of parameter settings to try as values.
        """
        if hpo_algotithm == HPOAlgorithm.GridSearch:
            param_config = [
                {
                    "alpha": [0.001, 0.01, 0.1, 1, 10, 100],
                    "rank_ratio": [0, 0.2, 0.4, 0.6, 0.8, 1],
                    "max_iter": [20, 50, 100, 200],
                    "optimizer": [
                        "avltree",
                        "direct-count",
                        "PRSVM",
                        "rbtree",
                        "simple",
                    ],
                }
            ]
        elif hpo_algotithm == HPOAlgorithm.RandomSearch:
            param_config = [
                {
                    "alpha": stats.uniform(0.001, 100),
                    "rank_ratio": stats.uniform(0, 1),
                    "max_iter": stats.randint(20, 200),
                    "optimizer": [
                        "avltree",
                        "direct-count",
                        "PRSVM",
                        "rbtree",
                        "simple",
                    ],
                }
            ]
        elif hpo_algotithm == HPOAlgorithm.BayesianSearch:
            param_config = [
                {
                    "alpha": Real(0.001, 100),
                    "rank_ratio": Real(0, 1),
                    "max_iter": Integer(20, 200),
                    "optimizer": [
                        "avltree",
                        "direct-count",
                        # "PRSVM",
                        "rbtree",
                        # "simple",
                    ],
                }
            ]
        return param_config
