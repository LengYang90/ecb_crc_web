from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.utils import resample
from .Data import DataBase, MultiOmicsData, BinaryData, ContinousData


class DataSamplerBase(ABC):
    """
    Base class for data samplers.
    """

    def __init__(
        self,
        data: Optional[DataBase] = None,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        random_state: Optional[int] = None,
    ):
        """
        Initialize the data sampler.

        Parameters:
        -----------
        data: Optional[DataBase]
            The data to be sampled, as a `DataBase` instance.
        X: Optional[pd.DataFrame]
            The input features, provided separately from `data`.
        y: Optional[pd.Series]
            The target variable, provided separately from `data`.
        random_state : Optional[int]
            Seed for random number generator.
        """
        if random_state is not None and (
            not isinstance(random_state, int) or random_state <= 0
        ):
            raise ValueError("random_state should be integer which larger than 0 !")
        # Check if data or X and y are provided and store the data
        self._is_data = self._check_data(data=data, X=X, y=y)
        self.data = data
        self.X = X
        self.y = y
        self.splits = None

        # Generate the initial train/test splits
        self.resplit(random_state=random_state)

    def _check_data(
        self,
        data: Optional[DataBase] = None,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
    ) -> bool:
        """
        Check if the appropriate combination of data, X, and y is provided.

        Parameters:
        -----------
        data : Optional[DataBase]
            The data to be sampled, as a `DataBase` instance.
        X : Optional[pd.DataFrame]
            The input features, provided separately from `data`.
        y : Optional[pd.Series]
            The target variable, provided separately from `data`.

        Returns:
        --------
        bool
            True if `data` is provided, False if `X` and `y` are provided.

        Raises:
        -------
        ValueError
            If neither `data` nor both `X` and `y` are provided, or if both are provided.
        """
        if data is None and (X is None or y is None):
            raise ValueError("You must provide either 'data' or both 'X' and 'y'.")

        if data is not None and (X is not None or y is not None):
            raise ValueError(
                "You cannot provide both 'data' and 'X'/'y'. Please provide either 'data' or both 'X' and 'y'."
            )

        if data is not None and not isinstance(data, DataBase):
            raise ValueError("data should be as a `DataBase` instance!")

        if X is not None and not isinstance(X, pd.DataFrame):
            raise ValueError("X should be as a `DataFrame` instance!")

        if y is not None and not isinstance(y, pd.Series):
            raise ValueError("X should be as a `Series` instance!")

        return data is not None

    @abstractmethod
    def resplit(self, random_state: Optional[int] = None):
        """
        Resplit the data into train/test sets.

        Parameters:
        -----------
        random_state : Optional[int]
            Seed for random number generator.
        """
        raise NotImplementedError("Subclasses should implement this!")

    def __len__(self) -> int:
        """
        Return the number of splits.

        Returns:
        --------
        int
            The number of splits.
        """
        return len(self.splits)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Get the train and test split for the given index.

        Parameters:
        -----------
        idx: int
            The index of the split.

        Returns:
        --------
        Tuple
            If `data` is provided, returns the train and test splits as `DataBase` instances.
            Otherwise, returns the train and test splits for `X` and `y` separately.
        """
        if idx >= len(self):
            raise IndexError("Index out of range")

        train_idx, test_idx = self.splits[idx]

        if self._is_data:
            return self.data[train_idx], self.data[test_idx]
        else:
            return (
                self.X.iloc[train_idx],
                self.X.iloc[test_idx],
                self.y.iloc[train_idx],
                self.y.iloc[test_idx],
            )


class DataSamplerSplit(DataSamplerBase):
    """
    Data sampler for a single train-test split.
    """

    def __init__(
        self,
        test_size: float,
        data: Optional[DataBase] = None,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        random_state: Optional[int] = None,
    ):
        """
        Initialize the data sampler for a single train-test split.

        Parameters:
        -----------
        test_size: float
            Proportion of the data to be used as test data.
        data: Optional[DataBase]
            The data to be sampled as a `DataBase` instance.
        X: Optional[pd.DataFrame]
            The input features, provided separately from `data`.
        y: Optional[pd.Series]
            The target variable, provided separately from `data`.
        random_state : Optional[int]
            Seed for the random number generator.
        """
        if not isinstance(test_size, float):
            raise ValueError("test_size in DataSamplerSplit should be float")
        if test_size <= 0 or test_size >= 1.0:
            raise ValueError("test_size in DataSamplerSplit should be in [0,1]")
        self.test_size = test_size
        super().__init__(data=data, X=X, y=y, random_state=random_state)

    def resplit(self, random_state: Optional[int] = None):
        """
        Resplit the data into a single train-test split.

        Parameters:
        -----------
        random_state : Optional[int]
            Seed for the random number generator.
        """

        self.random_state = random_state

        # Create a ShuffleSplit instance to generate the train-test split
        self.splitter = ShuffleSplit(
            n_splits=1, test_size=self.test_size, random_state=self.random_state
        )

        # Determine the number of samples based on whether `data` or `X`/`y` is used
        n = len(self.data) if self._is_data else len(self.X)

        # Generate the train-test splits and store them
        self.splits = list(self.splitter.split(range(n)))


class DataSamplerKFoldCV(DataSamplerBase):
    """
    Data sampler for K-Fold cross-validation.
    """

    def __init__(
        self,
        n_splits: int = 5,
        data: Optional[DataBase] = None,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        random_state: Optional[int] = None,
    ):
        """
        Initialize the data sampler for K-Fold cross-validation.

        Parameters:
        -----------
        n_splits: int, default=5
            The number of folds for cross-validation.
        data: Optional[DataBase]
            The data to be sampled as a `DataBase` instance.
        X: Optional[pd.DataFrame]
            The input features, provided separately from `data`.
        y: Optional[pd.Series]
            The target variable, provided separately from `data`.
        random_state: Optional[int]
            Seed for the random number generator.
        """
        if not isinstance(n_splits, int) or n_splits < 2:
            raise ValueError("n_splits should be integer which at least 2 !")
        self.n_splits = n_splits
        super().__init__(data=data, X=X, y=y, random_state=random_state)

    def resplit(self, random_state: Optional[int] = None):
        """
        Resplit the data into K folds.

        Parameters:
        -----------
        random_state: Optional[int]
            Seed for the random number generator.
        """
        self.random_state = random_state

        # Initialize the KFold splitter with shuffling enabled
        self.splitter = KFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

        # Determine the number of samples based on whether `data` or `X`/`y` is used
        n = len(self.data) if self._is_data else len(self.X)

        # Generate the train-test splits for K-Fold cross-validation
        self.splits = list(self.splitter.split(np.arange(n)))


class DataSamplerLOOCV(DataSamplerKFoldCV):
    """
    Data sampler for Leave-One-Out cross-validation.
    """

    def __init__(
        self,
        data: Optional[DataBase] = None,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        random_state: Optional[int] = None,
    ):
        """
        Initialize the data sampler for Leave-One-Out cross-validation.

        Parameters:
        -----------
        data: Optional[DataBase]
            The data to be sampled, as a `DataBase` instance.
        X: Optional[pd.DataFrame]
            The input features, provided separately from `data`.
        y: Optional[pd.Series]
            The target variable, provided separately from `data`.
        random_state: Optional[int]
            Seed for the random number generator.
        """
        # Check data and determine if it's a DataBase instance or separate X and y
        is_data = DataSamplerBase._check_data(self, data=data, X=X, y=y)

        # Initialize the parent class with the appropriate number of splits for LOOCV
        n_splits = len(data) if is_data else X.shape[0]
        super().__init__(
            data=data, X=X, y=y, n_splits=n_splits, random_state=random_state
        )


class DataSamplerBootstrap(DataSamplerBase):
    """
    Data sampler for bootstrap sampling.
    """

    def __init__(
        self,
        n_samples: int,
        test_size: float,
        data: Optional[DataBase] = None,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        feature_size: float = 1.0,
        feature_noise_gaussian: float = 0.05,
        feature_noise_bernoulli: float = 0.05,
        label_shuffle_size: float = 0.05,
        random_state: Optional[int] = None,
    ):
        """
        Initialize the data sampler for bootstrap sampling.

        Parameters:
        -----------
        n_samples: int
            Number of bootstrap samples to generate.
        test_size: float
            Proportion of the data to use as test data.
        feature_size: float
            Proportion of features to keep in bootstrap samples.
        feature_noise_gaussian: float
            Standard deviation of Gaussian noise added to features.
        feature_noise_bernoulli: float
            Proportion of features where Bernoulli noise is added.
        label_shuffle_size: float
            Proportion of labels to shuffle.
        random_state: Optional[int]
            Seed for random number generator.
        """
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer.")
        if not isinstance(test_size, float) or test_size <= 0 or test_size >= 1:
            raise ValueError("test_size must be a float between 0 and 1.")
        if not isinstance(feature_size, float) or feature_size <= 0 or feature_size > 1:
            raise ValueError("feature_size must be a float between 0 and 1.")
        if not isinstance(feature_noise_gaussian, float) or feature_noise_gaussian < 0:
            raise ValueError("feature_noise_gaussian must be a non-negative float.")
        if (
            not isinstance(feature_noise_bernoulli, float)
            or feature_noise_bernoulli < 0
        ):
            raise ValueError("feature_noise_bernoulli must be a non-negative float.")
        if (
            not isinstance(label_shuffle_size, float)
            or label_shuffle_size < 0
            or label_shuffle_size >= 1
        ):
            raise ValueError("label_shuffle_size must be a non-negative float be ")
        self.n_samples = n_samples
        self.test_size = test_size
        self.feature_size = feature_size
        self.feature_noise_gaussian = feature_noise_gaussian
        self.feature_noise_bernoulli = feature_noise_bernoulli
        self.label_shuffle_size = label_shuffle_size
        super().__init__(data=data, X=X, y=y, random_state=random_state)

    def resplit(self, random_state: Optional[int] = None):
        """
        Resplit the data using bootstrap sampling.

        Parameters:
        -----------
        random_state: Optional[int]
            Seed for random number generator to ensure reproducibility.
        """
        if random_state is not None:
            if not isinstance(random_state, int) or random_state < 0:
                raise ValueError("random_state must be None or a non-negative integer.")
        self.random_state = random_state
        data_len = len(self.data) if self._is_data else len(self.X)

        self.splits = []
        for i in range(self.n_samples):
            # Generate a unique random seed for each bootstrap sample
            seed = self.random_state + i if self.random_state is not None else None

            # Bootstrap sampling with replacement for training data
            train_len = int(data_len * (1 - self.test_size))
            train_idx = resample(
                np.arange(data_len),
                replace=True,
                n_samples=train_len,
                random_state=seed,
            )

            # Use remaining data points for test data
            test_idx = np.setdiff1d(np.arange(data_len), train_idx)

            self.splits.append((train_idx, test_idx))

    def __getitem__(self, idx: int) -> Tuple[
        Optional[pd.DataFrame],
        Optional[pd.DataFrame],
        Optional[pd.Series],
        Optional[pd.Series],
    ]:
        """
        Get the train and test split for the given index.

        Parameters:
        -----------
        idx: int
            The index of the split.

        Returns:
        --------
        Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series]]
            The train and test splits, which could include features and/or labels.
        """
        if not isinstance(idx, int):
            raise TypeError("Index must be an integer")
        if idx >= len(self):
            raise IndexError("Index out of range")

        train_idx, test_idx = self.splits[idx]

        random_state = (
            self.random_state + idx if self.random_state else self.random_state
        )

        if self._is_data:
            train_data = self.data[train_idx]
            test_data = self.data[test_idx]

            # Add noise and subset features for training data
            train_data = self._subset_features(train_data, random_state=random_state)
            train_data = self._add_label_noise(train_data, random_state=random_state)
            train_data = self._add_feature_noise(train_data, random_state=random_state)

            return train_data, test_data

        else:
            X_train = self.X.iloc[train_idx].copy()
            X_test = self.X.iloc[test_idx].copy()
            y_train = self.y.iloc[train_idx].copy()
            y_test = self.y.iloc[test_idx].copy()

            rng = np.random.default_rng(random_state)

            # subset features for training data
            feature_names = X_train.columns
            n_features_subset = int(len(feature_names) * self.feature_size)
            selected_features = rng.choice(
                feature_names, size=n_features_subset, replace=False
            )
            X_train = X_train[selected_features]
            X_test = X_test[selected_features]

            # Shuffle a proportion of the labels
            n_labels = len(y_train)
            n_to_shuffle = int(n_labels * self.label_shuffle_size)
            indices_to_shuffle = rng.choice(n_labels, size=n_to_shuffle, replace=False)
            y_train.iloc[indices_to_shuffle] = rng.permuted(
                np.copy(y_train)[indices_to_shuffle]
            )

            # Add Gaussian noise for regression labels
            if len(np.unique(y_train)) > 2 and self.feature_noise_gaussian > 0:
                noise = rng.normal(0, self.feature_noise_gaussian, y_train.shape)
                y_train += noise

            # Randomly flip binary feature values
            is_binary = X_train.apply(
                lambda col: len(np.unique(col)) == 2 and col.isin([0, 1]).all()
            )

            if sum(is_binary) > 0:
                noise = rng.binomial(
                    1, self.feature_noise_bernoulli, X_train.loc[:, is_binary].shape
                )
                X_train.loc[:, is_binary] = (
                    X_train.loc[:, is_binary].astype(int) ^ noise
                )  # XOR to flip bits where noise is 1

            # Add gaussian noise to continuous features
            if (sum(~is_binary)) > 0:
                noise = rng.normal(
                    0, self.feature_noise_gaussian, X_train.loc[:, ~is_binary].shape
                )
                X_train.loc[:, ~is_binary] += noise

            return X_train, X_test, y_train, y_test

    def _subset_features(self, data: DataBase, random_state=None) -> DataBase:
        """
        Randomly subset features based on feature_size.

        Parameters:
        -----------
        data : DataBase
            The data to be modified.
        random_state : Optional[int]
            Seed for random number generator to ensure reproducibility.

        Returns:
        --------
        DataBase
            The data with a subset of features.
        """
        # Check if data is an instance of DataBase
        if not isinstance(data, DataBase):
            raise TypeError("Input data must be an instance of DataBase")
        subset_data = data.copy()  # Create a copy to avoid modifying the original data

        # Seed for reproducibility in feature subsetting
        rng = np.random.default_rng(random_state)
        layer = "processed" if data.if_processed else "raw"

        if self.feature_size < 1.0:
            if isinstance(data, MultiOmicsData):
                to_update = {}
                for omics_name, omics_data in data.multi_omics_data.items():
                    feature_names = omics_data.get_feature_names()
                    n_features_subset = int(len(feature_names) * self.feature_size)
                    selected_features = rng.choice(
                        feature_names, size=n_features_subset, replace=False
                    )
                    to_update[omics_name] = omics_data[
                        :, selected_features
                    ].get_features(layer)
                    subset_data = subset_data.update_features(
                        **to_update, inplace=False
                    )
            else:
                feature_names = data.get_feature_names()
                n_features_subset = int(len(feature_names) * self.feature_size)
                selected_features = rng.choice(
                    feature_names, size=n_features_subset, replace=False
                )
                subset_data = subset_data[:, selected_features]

        return subset_data

    def _add_label_noise(self, data: DataBase, random_state=None) -> DataBase:
        """
        Add noise to the labels of the given data.

        Parameters:
        -----------
        data : DataBase
            The data to be modified.
        random_state : Optional[int]
            Seed for random number generator to ensure reproducibility.

        Returns:
        --------
        DataBase
            The data with noisy labels.
        """
        # Check if data is an instance of DataBase
        if not isinstance(data, DataBase):
            raise TypeError("Input data must be an instance of DataBase")

        # Create a copy to avoid modifying the original data
        noisy_data = data.copy()
        labels = data.get_labels().copy()
        label_type = data.get_label_type()

        # Seed for reproducibility in label noise addition
        rng = np.random.default_rng(random_state)


        # Shuffle a proportion of the labels
        if self.label_shuffle_size > 0:
            n_labels = len(labels)
            n_to_shuffle = int(n_labels * self.label_shuffle_size)
            indices_to_shuffle = rng.choice(n_labels, size=n_to_shuffle, replace=False)
            labels.iloc[indices_to_shuffle] = rng.permuted(
                np.copy(labels)[indices_to_shuffle]
            )

        # Add Gaussian noise for regression labels
        if label_type == "regression" and self.feature_noise_gaussian > 0:
            noise = rng.normal(0, self.feature_noise_gaussian, labels.shape)
            labels += noise

        return noisy_data.update_labels(labels, inplace=False)

    def _add_feature_noise(self, data: DataBase, random_state=None) -> DataBase:
        """
        Add noise to the features of the given data.

        Parameters:
        -----------
        data : DataBase
            The data to be modified.
        random_state : Optional[int]
            Seed for random number generator to ensure reproducibility.

        Returns:
        --------
        DataBase
            The data with noisy features.
        """
        # Check if data is an instance of DataBase
        if not isinstance(data, DataBase):
            raise ValueError("Input data must be an instance of DataBase.")
        # Check if random_state is None or a positive integer
        if random_state is not None and (
            not isinstance(random_state, int) or random_state < 0
        ):
            raise ValueError("random_state must be None or a positive integer.")
        noisy_data = data.copy()  # Create a copy to avoid modifying the original data
        to_update = {}
        layer = "processed" if data.if_processed else "raw"

        # Seed for reproducibility in feature noise addition
        rng = np.random.default_rng(random_state)

        if isinstance(noisy_data, MultiOmicsData):

            for omics_name, omics_data in noisy_data.multi_omics_data.items():
                features = omics_data.get_features(layer)
                if isinstance(omics_data, BinaryData):
                    if self.feature_noise_bernoulli > 0:
                        noise = rng.binomial(
                            1, self.feature_noise_bernoulli, omics_data.get_shape()
                        )
                        features ^= noise  # XOR to flip bits where noise is 1

                if isinstance(omics_data, ContinousData):
                    if self.feature_noise_gaussian > 0:
                        noise = rng.normal(
                            0, self.feature_noise_gaussian, omics_data.get_shape()
                        )
                        features += noise

                to_update[omics_name] = features
            noisy_data.update_features(**to_update, inplace=False)
        else:
            features = noisy_data.get_features(layer)
            if isinstance(noisy_data, BinaryData):
                if self.feature_noise_bernoulli > 0:
                    noise = rng.binomial(
                        1, self.feature_noise_bernoulli, noisy_data.get_shape()
                    )
                    features ^= noise  # XOR to flip bits where noise is 1

            if isinstance(noisy_data, ContinousData):
                if self.feature_noise_gaussian > 0:
                    noise = rng.normal(
                        0, self.feature_noise_gaussian, noisy_data.get_shape()
                    )
                    features += noise
            noisy_data = noisy_data.update_features(features, inplace=False)

        return noisy_data