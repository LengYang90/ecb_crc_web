from __future__ import annotations
from typing import Optional
from collections import OrderedDict


import pandas as pd
import numpy as np

# from MLGenie.design.Utils import read_gene_name_mapper


class DataBase(object):
    """
    Base class for MLGenie Data

    #TODO: Merge different batches of data.
    """

    def __init__(
        self,
        features: Union[pd.DataFrame, List[pd.DataFrame]],
        labels: Union[pd.Series, List[pd.Series]] = None,
        if_processed: bool = False,
        feature_names=None,
        sample_names=None,
    ):
        """
        Initializes the DataBase with features, labels, feature names, and sample names.
        If `if_processed` is True, features are considered processed and further processing is locked.
        If `if_processed` is False, processed features can be assigned later.

        Parameters:
            features (Union[pd.DataFrame, List[pd.DataFrame]]): The feature matrix.
            labels (Union[pd.Series, List[pd.Series]], optional): The labels of the dataset.
            if_processed (bool): Flag indicating if the features are already processed.
            feature_names (array-like, optional): The feature names. If None, inferred from features columns.
            sample_names (array-like, optional): The sample names. If None, inferred from features index.
        """
        features, labels = self._check_data_validity(features, labels)

        self.features = {
            "raw": features,
            "processed": features if if_processed else None,
        }

        self.labels = labels
        self.if_processed = if_processed

        self.feature_names = (
            pd.Index(feature_names) if feature_names is not None else features.columns
        )
        self.sample_names = (
            pd.Index(sample_names) if sample_names is not None else features.index
        )
        self.label_type = self._check_label_type()

    def get_shape(self):
        """
        Get the shape of the feature matrix.

        Returns:
            tuple: Shape of the feature matrix.
        """
        if self.if_processed:
            return self.features["processed"].shape
        else:
            return self.features["raw"].shape

    def get_features(self, layer: str, selected_features=None) -> pd.DataFrame:
        """
        Get the features of data based on the specified layer.

        Parameters:
            layer (str): The layer to retrieve features from. Must be 'raw' or 'processed'.
            selected_features(list, or pd.series): The index of selected features names.

        Returns:
            pd.DataFrame: The feature matrix for the specified layer.
        """
        if layer in self.features:
            features = self.features[layer]
            if selected_features is None:
                return features
            else:
                if features is None:
                    raise ValueError(
                        "processed features is None, can not get selected_features"
                    )
                return pd.DataFrame(features[selected_features])
        else:
            raise ValueError("Invalid layer specified. Must be 'raw' or 'processed'.")

    def get_labels(self) -> pd.Series:
        """
        Get the labels of data.

        Returns:
            pd.Series: The labels of the dataset.
        """
        return self.labels

    def get_label_type(self) -> pd.Series:
        """
        Get the type of label.

        Returns:
            string: The label type.
        """
        return self.label_type

    def get_feature_names(self) -> pd.Index:
        """
        Get the feature names.

        Returns:
            pd.Index: The feature names.
        """
        return self.feature_names

    def get_sample_names(self) -> pd.Index:
        """
        Get the sample names.

        Returns:
            pd.Index: The sample names.
        """
        return self.sample_names

    def update_features(
        self,
        features: Union[pd.DataFrame, List[pd.DataFrame]],
        layer: str = "processed",
        inplace: bool = True,
    ) -> DataBase:
        """
        Update the specified layer with the given features

        Parameters:
            features (pd.DataFrame): The feature matrix.
            layer (str): The layer to retrieve features from. Must be 'raw' or 'processed'.
            inplace (bool, optional): If True, the `DataBase` object is updated in-place. If False, a new object is returned.
        Returns:
            DataBase: A new or updated DataBase object
        """
        if layer not in self.features:
            raise ValueError("Invalid layer specified. Must be 'raw' or 'processed'.")

        features, labels = self._check_data_validity(
            features=features, labels=self.labels
        )

        if inplace:
            self.features[layer] = features
            self.feature_names = features.columns
            self.sample_names = features.index
            return self
        else:
            return self.__class__(
                features=features,
                labels=self.labels,
                if_processed=self.if_processed,
                feature_names=features.columns,
                sample_names=features.index,
            )

    def update_processed_flag(self, inplace: bool = True) -> DataBase:
        """
        After all preprocessing is completed, if_processed is set to True

        Parameters:
            inplace (bool, optional): If True, the `DataBase` object is updated in-place. If False, a new object is returned.
        Returns:
            DataBase: A new or updated DataBase object
        """
        if self.features["processed"] is None:
            raise ValueError(
                'features["processed"] is None! Can not update processed flag'
            )
        if inplace:
            self.if_processed = True
            return self
        else:
            layer = "processed" if self.if_processed else "raw"
            return self.__class__(
                features=self.get_features(layer),
                labels=self.labels,
                if_processed=True,
                feature_names=self.feature_names,
                sample_names=self.sample_names,
            )

    def update_labels(
        self, labels: Union[pd.Series, List[pd.Series]], inplace: bool = True
    ) -> DataBase:
        """
        Update the specified layer with the given features

        Parameters:
            labels (pd.Series): The label of the data.
            inplace (bool, optional): If True, the `DataBase` object is updated in-place. If False, a new object is returned.
        Returns:
            DataBase: A new or updated DataBase object
        """
        layer = "processed" if self.if_processed else "raw"
        features, labels = self._check_data_validity(
            features=self.get_features(layer), labels=labels
        )

        if inplace:
            self.labels = labels
            return self
        else:
            return self.__class__(
                features=self.get_features(layer),
                labels=labels,
                if_processed=self.if_processed,
                feature_names=self.feature_names,
                sample_names=self.sample_names,
            )

    def copy(self) -> DataBase:
        """
        Copy of DataBase object

        Returns:
            DataBase: A new DataBase object
        """
        layer = "processed" if self.if_processed else "raw"
        return self.__class__(
            features=self.get_features(layer),
            labels=self.labels,
            if_processed=self.if_processed,
            feature_names=self.feature_names,
            sample_names=self.sample_names,
        )

    def _check_feature_validity(
        self, features: Union[pd.DataFrame, List[pd.DataFrame]]
    ) -> pd.DataFrame:
        """
        Checks the validity of the features.

        Parameters:
            features (Union[pd.DataFrame, List[pd.DataFrame]]): The feature matrix.

        Returns:
            features: The feature matrix.
        """
        assert isinstance(features, pd.DataFrame) or isinstance(
            features, list
        ), "Features should be DataFrame type or a list of DataFrame!"
        if isinstance(features, list):
            assert len(features) > 0, "Features list is empty!"
            for feat in features:
                assert isinstance(
                    feat, pd.DataFrame
                ), "Each item in feature list should be DataFrame type"
            features = pd.concat(features, axis=0, join="outer")
        assert not features.empty, "Features is empty!"

        try:
            # Filter out non-numeric features.
            features = features.select_dtypes(include=[int, float, bool])
        except ValueError as e:
            raise ValueError("Failed to filter out non-numeric features: " + str(e))
        return features

    def _check_label_validity(
        self, labels: Union[pd.Series, List[pd.Series]]
    ) -> pd.Series:
        """
        Checks the validity of the features.

        Parameters:
            labels (Union[pd.Series, List[pd.Series]], optional): The labels of the dataset.

        Returns:
            labels: The labels.
        """
        assert (
            isinstance(labels, pd.Series) or labels is None or isinstance(labels, list)
        ), "Labels shold be pd.Series type!"
        if isinstance(labels, list):
            assert len(labels) > 0, "Labels list is empty!"
            for label in labels:
                assert isinstance(
                    label, pd.Series
                ), "Each item in label list should be Series type"
            labels = pd.concat(labels, axis=0, join="outer")
        return labels

    def _check_data_validity(
        self,
        features: Union[pd.DataFrame, List[pd.DataFrame]],
        labels: Union[pd.Series, List[pd.Series]] = None,
    ) -> tuple:
        """
        Checks the validity of the data by ensuring no duplicates.

        Parameters:
            features (Union[pd.DataFrame, List[pd.DataFrame]]): The feature matrix.
            labels (Union[pd.Series, List[pd.Series]], optional): The labels of the dataset.

        Returns:
            tuple: The feature matrix and the labels.
        """
        features = self._check_feature_validity(features)
        labels = self._check_label_validity(labels)

        features = self._drop_duplicates(features)
        if labels is not None:
            common_index = labels.index.intersection(features.index)
            labels = labels.loc[common_index]  # .dropna()
            features = features.loc[common_index]
            if set(features.index) != set(labels.index):
                diff = pd.Index.difference(features.index, labels.index)
                raise ValueError(
                    "Features index not equal to labels index, difference are [ {} ]".format(
                        ", ".join([str(ind) for ind in diff])
                    )
                )

        return features, labels

    def _check_label_type(self) -> str:
        """
        check the type of label, and assign to self.label_type, can be "classification", "regression" or None

        Returns:
            str: The label type
        """
        if self.labels is None:
            return None
        num_unique_values = self.labels.nunique()
        if num_unique_values > 2:
            return "regression"
        else:
            return "classification"

    def _drop_duplicates(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Drop duplicated feature names and sample ids
        """
        dup_index = features.index.to_series().duplicated()
        dup_column = features.columns.to_series().duplicated()

        return features.loc[~dup_index, ~dup_column]

    def __len__(self) -> int:
        """
        Get the length of the data.

        Returns:
            int: The length of the data.
        """
        return self.get_shape()[0]

    def __getitem__(self, index) -> DataBase:
        """
        Slice or index the data and return a new DataBase instance with the sliced data.

        Parameters:
            index (slice, list, tuple, or array-like): The index or slice to apply. Supports both row and column indexing.

        Returns:
            DataBase: A new DataBase object containing the sliced data.
        """
        if isinstance(index, tuple):
            row_idx, col_idx = index
        else:
            row_idx, col_idx = index, slice(None)

        features = (
            self.features["processed"] if self.if_processed else self.features["raw"]
        )
        if isinstance(row_idx, str):
            sliced_features = pd.DataFrame(features.loc[row_idx]).transpose()
            sliced_labels = (
                pd.Series([self.labels.loc[row_idx]], index=sliced_features.index)
                if self.labels is not None
                else None
            )

        elif isinstance(row_idx, int):
            sliced_features = pd.DataFrame(features.iloc[row_idx, col_idx]).transpose()
            sliced_labels = (
                pd.Series([self.labels.iloc[row_idx]], index=sliced_features.index)
                if self.labels is not None
                else None
            )

        elif isinstance(row_idx, (pd.Series, list)):#, np.ndarray
            row_idx = pd.Series(row_idx)
            if not row_idx.isnull().any():
                row_idx = index
                sliced_features = features.loc[row_idx]
                sliced_labels = (
                    self.labels.loc[row_idx] if self.labels is not None else None
                )

            else:
                # if target index is not exist in original index, add a NA row
                sliced_features = []
                sliced_labels = []
                for idx, value in row_idx.items():
                    if pd.isnull(value):
                        na_features = [[np.nan] * self.get_shape()[1]]
                        sliced_features.append(np.array(na_features))
                        if self.labels is not None:
                            sliced_labels.append(np.nan)
                    else:
                        sliced_features.append(
                            pd.DataFrame(features.loc[value]).transpose().values
                        )
                        if self.labels is not None:
                            sliced_labels.append(self.labels.loc[value])

                sliced_features = np.concatenate(sliced_features, axis=0)
                sliced_features = pd.DataFrame(
                    sliced_features, index=row_idx.index, columns=self.feature_names
                )
                sliced_labels = (
                    pd.Series(sliced_labels, index=row_idx.index)
                    if self.labels is not None
                    else None
                )

        else:
            sliced_features = features.iloc[row_idx, col_idx]
            sliced_labels = (
                self.labels.iloc[row_idx] if self.labels is not None else None
            )
        # Rename duplicate index of sliced_features
        if not sliced_features.index.is_unique:
            sliced_features.index = pd.Index(
                [
                    f"{idx}_{i}" if sliced_features.index.duplicated()[i] else idx
                    for i, idx in enumerate(sliced_features.index)
                ]
            )
            if sliced_labels is not None:
                sliced_labels.index = sliced_features.index

        return self.__class__(
            features=pd.DataFrame(sliced_features),
            labels=pd.Series(sliced_labels),
            if_processed=self.if_processed,
        )


class OmicsData(DataBase):
    """
    Class to handle omics data, supporting different gene naming systems and organism types.
    """

    def __init__(
        self,
        features: Union[pd.DataFrame, List[pd.DataFrame]],
        labels: Union[pd.Series, List[pd.Series]] = None,
        organism: str = "human",
        if_processed: bool = False,
        feature_names=None,
        sample_names=None,
    ):
        """
        Initializes the OmicsData with features, labels, and organism type.

        Parameters:
            features (Union[pd.DataFrame, List[pd.DataFrame]]): The feature matrix.
            labels (Union[pd.Series, List[pd.Series]], optional): The labels of the dataset.
            organism (str): The organism type, should be one of ['human', 'mouse', 'yeast'].
            if_processed (bool): Flag indicating if the features are already processed.
            feature_names (array-like, optional): The feature names. If None, inferred from features columns.
            sample_names (array-like, optional): The sample names. If None, inferred from features index.
        """
        super().__init__(
            features=features,
            labels=labels,
            if_processed=if_processed,
            feature_names=feature_names,
            sample_names=sample_names,
        )
        self.organism = organism
        assert organism in ["human", "mouse", "yeast"], "Organism not supported!"

        # Decode feature names if needed
        # self._decode_feature_names()

    def get_feature_names(self, name=None):
        """
        Get the feature names based on the specified naming system.

        Parameters:
            name (str, optional): The naming system to use. Must be one of ['hgnc', 'ensemble'].

        Returns:
            pd.Index: The feature names.
        """
        if name is None:
            return super().get_feature_names()
        assert name in ["hgnc", "ensemble"], "Gene name not supported!"
        return self._feature_name[name]

    def _decode_feature_names(self):
        """
        Decode and translate feature names to different gene naming systems based on the organism.
        """
        gene_name_mapper = read_gene_name_mapper(self.organism)
        self._feature_name = pd.DataFrame(
            {
                "raw": self.features["raw"].columns,
                "hgnc": gene_name_mapper.get_hgnc_names(self.features["raw"].columns),
                "ensemble": gene_name_mapper.get_ensemble_names(
                    self.features["raw"].columns
                ),
            }
        )

        # Ensure that all columns are populated
        assert all(
            col in self._feature_name.columns for col in ["raw", "hgnc", "ensemble"]
        ), "Feature names not fully mapped."


class BinaryData(OmicsData):
    """
    Class to handle binary data
    """

    def __init__(
        self,
        features: Union[pd.DataFrame, List[pd.DataFrame]],
        labels: Union[pd.Series, List[pd.Series]] = None,
        organism: str = "human",
        if_processed=False,
        feature_names=None,
        sample_names=None,
    ):
        """
        Initializes the BinaryData with features, labels, and organism type.

        Parameters:
            features (Union[pd.DataFrame, List[pd.DataFrame]]): The feature matrix.
            labels (Union[pd.Series, List[pd.Series]], optional): The labels of the dataset.
            organism (str): The organism type, should be one of ['human', 'mouse', 'yeast'].
            if_processed (bool): Flag indicating if the features are already processed.
            feature_names (array-like, optional): The feature names. If None, inferred from features columns.
            sample_names (array-like, optional): The sample names. If None, inferred from features index.
        """
        self._check_binary_validity(features, if_processed)

        super().__init__(
            features=features,
            labels=labels,
            organism=organism,
            if_processed=if_processed,
            feature_names=feature_names,
            sample_names=sample_names,
        )

    def _check_binary_validity(
        self, features: Union[pd.DataFrame, List[pd.DataFrame]], if_processed: bool
    ) -> tuple:
        """
        Checks the validity of the binary data by ensuring all values are 0, 1 or NA.
        """
        features = super()._check_feature_validity(features)

        try:
            features = features.astype(int)
        except ValueError as e:
            raise ValueError("Failed to covert the binary features to int type: " + str(e))

        # Check for invalid values in the features
        if not if_processed and not (features.isin([0, 1]) | features.isna()).all().all():
            raise ValueError("Binary data must only contain 0, 1, or NA values.")


class ContinousData(OmicsData):
    """
    Class to handle continous data
    """

    def __init__(
        self,
        features: Union[pd.DataFrame, List[pd.DataFrame]],
        labels: Union[pd.Series, List[pd.Series]] = None,
        organism: str = "human",
        if_processed=False,
        feature_names=None,
        sample_names=None,
    ):
        """
        Initializes the ContinousData with features, labels, and organism type.

        Parameters:
            features (Union[pd.DataFrame, List[pd.DataFrame]]): The feature matrix.
            labels (Union[pd.Series, List[pd.Series]], optional): The labels of the dataset.
            organism (str): The organism type, should be one of ['human', 'mouse', 'yeast'].
            if_processed (bool): Flag indicating if the features are already processed.
            feature_names (array-like, optional): The feature names. If None, inferred from features columns.
            sample_names (array-like, optional): The sample names. If None, inferred from features index.
        """
        self._check_continous_validity(features)

        super().__init__(
            features=features,
            labels=labels,
            organism=organism,
            if_processed=if_processed,
            feature_names=feature_names,
            sample_names=sample_names,
        )

    def _check_continous_validity(
        self,
        features: Union[pd.DataFrame, List[pd.DataFrame]],
    ):
        """
        Checks the validity of the continous data by ensuring all values are float or NA.
        """
        features = super()._check_feature_validity(features)

        try:
            features = features.astype(float)
        except ValueError as e:
            raise ValueError("Failed to covert the continuous features to float type: " + str(e))


class TimeSeriesData(ContinousData):
    """
    Class to time series data, supporting different gene naming systems and organism types.
    """

    def __init__(
        self,
        features: Union[pd.DataFrame, List[pd.DataFrame]],
        labels: Union[pd.Series, List[pd.Series]] = None,
        organism: str = "human",
        if_processed: bool = False,
        feature_names=None,
        sample_names=None,
    ):
        """
        Initializes the TimeSeriesData with features, labels, and organism type.

        Parameters:
            features (Union[pd.DataFrame, List[pd.DataFrame]]): The feature matrix containing gene mutation data.
            labels (Union[pd.Series, List[pd.Series]], optional): The labels of the dataset.
            organism (str): The organism type, should be one of ['human', 'mouse', 'yeast'].
            if_processed (bool): Flag indicating if the features are already processed.
            feature_names (array-like, optional): The feature names. If None, inferred from features columns.
            sample_names (array-like, optional): The sample names. If None, inferred from features index.
        """

        super().__init__(
            features=features,
            labels=labels,
            organism=organism,
            if_processed=if_processed,
            feature_names=feature_names,
            sample_names=sample_names,
        )


class MetaData(OmicsData):
    """
    Class to handle meta data, supporting different gene naming systems and organism types.
    """

    def __init__(
        self,
        features: Union[pd.DataFrame, List[pd.DataFrame]],
        labels: Union[pd.Series, List[pd.Series]] = None,
        organism: str = "human",
        if_processed: bool = False,
        feature_names=None,
        sample_names=None,
    ):
        """
        Initializes the MetaData with features, labels, and organism type.

        Parameters:
            features (Union[pd.DataFrame, List[pd.DataFrame]]): The feature matrix containing gene mutation data.
            labels (Union[pd.Series, List[pd.Series]], optional): The labels of the dataset.
            organism (str): The organism type, should be one of ['human', 'mouse', 'yeast'].
            if_processed (bool): Flag indicating if the features are already processed.
            feature_names (array-like, optional): The feature names. If None, inferred from features columns.
            sample_names (array-like, optional): The sample names. If None, inferred from features index.
        """
        super().__init__(
            features=features,
            labels=labels,
            organism=organism,
            if_processed=if_processed,
            feature_names=feature_names,
            sample_names=sample_names,
        )


class GeneMutationData(BinaryData, OmicsData):
    """
    Handles gene mutation data, ensuring all features are binary or NA.
    """

    def __init__(
        self,
        features: Union[pd.DataFrame, List[pd.DataFrame]],
        labels: Union[pd.Series, List[pd.Series]] = None,
        organism="human",
        if_processed=False,
        feature_names=None,
        sample_names=None,
    ):
        """
        Initializes the GeneMutationData with features, labels, and organism type.

        Parameters:
            features (Union[pd.DataFrame, List[pd.DataFrame]]): The feature matrix containing gene mutation data.
            labels (Union[pd.Series, List[pd.Series]], optional): The labels of the dataset.
            organism (str): The organism type, should be one of ['human', 'mouse', 'yeast'].
            if_processed (bool): Flag indicating if the features are already processed.
            feature_names (array-like, optional): The feature names. If None, inferred from features columns.
            sample_names (array-like, optional): The sample names. If None, inferred from features index.
        """
        super().__init__(
            features=features,
            labels=labels,
            organism=organism,
            if_processed=if_processed,
            feature_names=feature_names,
            sample_names=sample_names,
        )


class GeneExpressionData(ContinousData, OmicsData):
    """
    Handles gene expression data, ensuring all features are contious float or NA.
    """

    def __init__(
        self,
        features: Union[pd.DataFrame, List[pd.DataFrame]],
        labels: Union[pd.Series, List[pd.Series]] = None,
        organism: str = "human",
        if_processed: bool = False,
        count: bool = False,
        feature_names=None,
        sample_names=None,
    ):
        """
        Initializes the GeneExpressionData with features, labels, and organism type.

        Parameters:
            features (Union[pd.DataFrame, List[pd.DataFrame]]): The feature matrix containing gene mutation data.
            labels (Union[pd.Series, List[pd.Series]]): The labels of the dataset.
            organism (str): The organism type, should be one of ['human', 'mouse', 'yeast'].
            if_processed (bool): Flag indicating if the features are already processed.
            count(bool): Flag indicating if the features are read count.
            feature_names (array-like, optional): The feature names. If None, inferred from features columns.
            sample_names (array-like, optional): The sample names. If None, inferred from features index.
        """
        self.count = count

        super().__init__(
            features=features,
            labels=labels,
            organism=organism,
            if_processed=if_processed,
            feature_names=feature_names,
            sample_names=sample_names,
        )

    def _transform_data(
        self, features: Union[pd.DataFrame, List[pd.DataFrame]]
    ) -> pd.DataFrame:
        """
        If the input is read count, apply log2(v+1) transformation, otherwise not.

        Parameters:
            features (pd.DataFrame): The feature matrix containing gene mutation data.

        Returns:
            pd.DataFrame: The transformed features
        """
        features = super()._check_feature_validity(features)

        if self.count:
            if features.dtypes.eq("int64").all() and features.min().min() >= 0:
                features = features.apply(lambda v: np.log2(v + 1))
        return features


class ProteinExpressionData(ContinousData, OmicsData):
    """
    Handles protein expression data, ensuring all features are contious float or NA.
    """

    def __init__(
        self,
        features: Union[pd.DataFrame, List[pd.DataFrame]],
        labels: Union[pd.Series, List[pd.Series]] = None,
        organism="human",
        if_processed=False,
        feature_names=None,
        sample_names=None,
    ):
        """
        Initializes the ProteinExpressionData with features, labels, and organism type.

        Parameters:
            features (Union[pd.DataFrame, List[pd.DataFrame]]): The feature matrix containing gene mutation data.
            labels (Union[pd.Series, List[pd.Series]], optional): The labels of the dataset.
            organism (str): The organism type, should be one of ['human', 'mouse', 'yeast'].
            if_processed (bool): Flag indicating if the features are already processed.
            feature_names (array-like, optional): The feature names. If None, inferred from features columns.
            sample_names (array-like, optional): The sample names. If None, inferred from features index.
        """
        super().__init__(
            features=features,
            labels=labels,
            organism=organism,
            if_processed=if_processed,
            feature_names=feature_names,
            sample_names=sample_names,
        )


class GeneCNVData(ContinousData, OmicsData):
    """
    Handles gene CNV data, ensuring all features are contious float or NA.
    """

    def __init__(
        self,
        features: Union[pd.DataFrame, List[pd.DataFrame]],
        labels: Union[pd.Series, List[pd.Series]] = None,
        organism="human",
        if_processed=False,
        feature_names=None,
        sample_names=None,
    ):
        """
        Initializes the GeneCNVData with features, labels, and organism type.

        Parameters:
            features (Union[pd.DataFrame, List[pd.DataFrame]]): The feature matrix containing gene mutation data.
            labels (Union[pd.Series, List[pd.Series]], optional): The labels of the dataset.
            organism (str): The organism type, should be one of ['human', 'mouse', 'yeast'].
            if_processed (bool): Flag indicating if the features are already processed.
            feature_names (array-like, optional): The feature names. If None, inferred from features columns.
            sample_names (array-like, optional): The sample names. If None, inferred from features index.
        """
        features = self._transform_data(features)

        super().__init__(
            features=features,
            labels=labels,
            organism=organism,
            if_processed=if_processed,
            feature_names=feature_names,
            sample_names=sample_names,
        )

    def _transform_data(
        self, features: Union[pd.DataFrame, List[pd.DataFrame]]
    ) -> pd.DataFrame:
        """
        If original copy number (all >= 0), apply (value - 2) to convert to  copy number changes, then truncate all value to (min, max) e.g. (-5, 5)

        Parameters:
            features (Union[pd.DataFrame, List[pd.DataFrame]]): The feature matrix containing gene mutation data.

        Returns:
            pd.DataFrame: The transformed features
        """
        features = super()._check_feature_validity(features)

        if features.min().min() >= 0:
            features = (features - 2).clip(lower=-5.0, upper=5.0)
        return features


class GeneMethylationData(ContinousData, OmicsData):
    """
    Handles gene methylation data, ensuring all features are between 0 and 1  or NA.
    """

    def __init__(
        self,
        features: Union[pd.DataFrame, List[pd.DataFrame]],
        labels: Union[pd.Series, List[pd.Series]] = None,
        organism: str = "human",
        if_processed: bool = False,
        feature_names=None,
        sample_names=None,
    ):
        """
        Initializes the GeneMethylationData with features, labels, and organism type.

        Parameters:
            features (Union[pd.DataFrame, List[pd.DataFrame]]): The feature matrix containing gene mutation data.
            labels (Union[pd.Series, List[pd.Series]]): The labels of the dataset.
            organism (str): The organism type, should be one of ['human', 'mouse', 'yeast'].
            if_processed (bool): Flag indicating if the features are already processed.
            feature_names (array-like, optional): The feature names. If None, inferred from features columns.
            sample_names (array-like, optional): The sample names. If None, inferred from features index.
        """
        self.check_contious_range_validity(features, if_processed)

        super().__init__(
            features=features,
            labels=labels,
            organism=organism,
            if_processed=if_processed,
            feature_names=feature_names,
            sample_names=sample_names,
        )

    def check_contious_range_validity(
        self,
        features: Union[pd.DataFrame, List[pd.DataFrame]],
        if_processed: bool,
    ) -> tuple:
        """
        Checks the validity of the continous data by ensuring all values are between [0,1].
        """
        features = super()._check_feature_validity(features)

        # Check for invalid values in the features
        if not if_processed and not ((features >= 0) & (features <= 1) | features.isna()).all().all():
            raise ValueError("Gene Methylation Data must between [0,1]")


class MultiOmicsData(DataBase):
    """
    A class to handle multi-omics data by combining different omics data types.
    """

    def __init__(
        self,
        gene_mutation: Optional[pd.DataFrame, List[pd.DataFrame]] = None,
        gene_expression: Optional[pd.DataFrame, List[pd.DataFrame]] = None,
        gene_fusion: Optional[pd.DataFrame, List[pd.DataFrame]] = None,
        gene_cnv: Optional[pd.DataFrame, List[pd.DataFrame]] = None,
        gene_methylation: Optional[pd.DataFrame, List[pd.DataFrame]] = None,
        mirna_expression: Optional[pd.DataFrame, List[pd.DataFrame]] = None,
        circrna_expression: Optional[pd.DataFrame, List[pd.DataFrame]] = None,
        protein_expression: Optional[pd.DataFrame, List[pd.DataFrame]] = None,
        protein_phosphorylation: Optional[pd.DataFrame, List[pd.DataFrame]] = None,
        protein_acetylation: Optional[pd.DataFrame, List[pd.DataFrame]] = None,
        time_series_data: Optional[pd.DataFrame, List[pd.DataFrame]] = None,
        meta_data: Optional[pd.DataFrame, List[pd.DataFrame]] = None,
        scrna_data: Optional[pd.DataFrame, List[pd.DataFrame]] = None,
        metabolomics_data: Optional[pd.DataFrame, List[pd.DataFrame]] = None,
        other_data: Optional[pd.DataFrame, List[pd.DataFrame]] = None,
        labels: Optional[Union[pd.Series, List[pd.Series]]] = None,
        organism: str = "human",
        if_processed: bool = False,
        concat_mode: str = "inner",
    ):
        """
        Initializes the MultiOmicsData by creating individual omics as and concatenating their features.

        Parameters:
            gene_mutation (pd.DataFrame, optional): Gene mutation data.
            gene_expression (pd.DataFrame, optional): Gene expression data.
            gene_fusion (pd.DataFrame, optional): Gene fusion data.
            gene_cnv (pd.DataFrame, optional): Gene CNV data.
            gene_methylation (pd.DataFrame, optional): Gene methylation data.
            mirna_expression (pd.DataFrame, optional): miRNA expression data.
            circrna_expression (pd.DataFrame, optional): circRNA expression data.
            protein_expression (pd.DataFrame, optional): Protein expression data.
            protein_phosphorylation (pd.DataFrame, optional): Protein phosphorylation data.
            protein_acetylation (pd.DataFrame, optional): Protein acetylation data.
            time_series_data (pd.DataFrame, optional): Time-series data.
            meta_data (pd.DataFrame, optional): Metadata.
            scrna_data (pd.DataFrame, optional): Single-cell RNA data.
            metabolomics_data (pd.DataFrame, optional): Metabolomics data.
            other_data (pd.DataFrame, optional): Other data.
            labels (pd.Series, optional): Labels for the dataset.
            organism (str): Organism type.
            if_processed (bool): Flag to indicate if data is processed.
            concat_mode(str): The mode of concat features, must be one of ["inner", "outer"]
        """
        # Initialize individual omics classes
        self.gene_mutation = (
            None
            if (isinstance(gene_mutation, list) and len(gene_mutation) == 0) or gene_mutation is None
            else GeneMutationData(
                features=gene_mutation,
                labels=labels,
                organism=organism,
                if_processed=if_processed,
            )
        )

        self.gene_expression = (
            None
            if (isinstance(gene_expression, list) and len(gene_expression) == 0) or gene_expression is None
            else GeneExpressionData(
                features=gene_expression,
                labels=labels,
                organism=organism,
                if_processed=if_processed,
            )
        )

        self.gene_fusion = (
            None
            if (isinstance(gene_fusion, list) and len(gene_fusion) == 0) or gene_fusion is None
            else GeneExpressionData(
                features=gene_fusion,
                labels=labels,
                organism=organism,
                if_processed=if_processed,
            )
        )

        self.gene_cnv = (
            None
            if (isinstance(gene_cnv, list) and len(gene_cnv) == 0) or gene_cnv is None
            else GeneCNVData(
                features=gene_cnv,
                labels=labels,
                organism=organism,
                if_processed=if_processed,
            )
        )

        self.gene_methylation = (
            None
            if (isinstance(gene_methylation, list) and len(gene_methylation) == 0) or gene_methylation is None
            else GeneMethylationData(
                features=gene_methylation,
                labels=labels,
                organism=organism,
                if_processed=if_processed,
            )
        )

        self.mirna_expression = (
            None
            if (isinstance(mirna_expression, list) and len(mirna_expression) == 0) or mirna_expression is None
            else GeneExpressionData(
                features=mirna_expression,
                labels=labels,
                organism=organism,
                if_processed=if_processed,
            )
        )
        self.circrna_expression = (
            None
            if (isinstance(circrna_expression, list) and len(circrna_expression) == 0) or circrna_expression is None
            else GeneExpressionData(
                features=circrna_expression,
                labels=labels,
                organism=organism,
                if_processed=if_processed,
            )
        )
        self.protein_expression = (
            None
            if (isinstance(protein_expression, list) and len(protein_expression) == 0) or protein_expression is None
            else ProteinExpressionData(
                features=protein_expression,
                labels=labels,
                organism=organism,
                if_processed=if_processed,
            )
        )
        self.protein_phosphorylation = (
            None
            if (isinstance(protein_phosphorylation, list) and len(protein_phosphorylation) == 0) or protein_phosphorylation is None
            else  ProteinExpressionData(
                features=protein_phosphorylation,
                labels=labels,
                organism=organism,
                if_processed=if_processed,
            )
        )
        self.protein_acetylation = (
            None
            if (isinstance(protein_acetylation, list) and len(protein_acetylation) == 0) or protein_acetylation is None
            else ProteinExpressionData(
                features=protein_acetylation,
                labels=labels,
                organism=organism,
                if_processed=if_processed,
            )
        )
        self.time_series_data = (
            None
            if (isinstance(time_series_data, list) and len(time_series_data) == 0) or time_series_data is None
            else TimeSeriesData(
                features=time_series_data,
                labels=labels,
                organism=organism,
                if_processed=if_processed,
            )
        )
        self.meta_data = (
            None
            if (isinstance(meta_data, list) and len(meta_data) == 0) or meta_data is None
            else MetaData(
                features=meta_data,
                labels=labels,
                organism=organism,
                if_processed=if_processed,
            )
        )
        self.scrna_data = (
            None
            if (isinstance(scrna_data, list) and len(scrna_data) == 0) or scrna_data is None
            else MetaData(
                features=scrna_data,
                labels=labels,
                organism=organism,
                if_processed=if_processed,
            )
        )
        self.metabolomics_data = (
            None
            if (isinstance(metabolomics_data, list) and len(metabolomics_data) == 0) or metabolomics_data is None
            else MetaData(
                features=metabolomics_data,
                labels=labels,
                organism=organism,
                if_processed=if_processed,
            )
        )
        self.other_data = (
            None
            if (isinstance(other_data, list) and len(other_data) == 0) or other_data is None
            else MetaData(
                features=other_data,
                labels=labels,
                organism=organism,
                if_processed=if_processed,
            )
        )

        self.if_processed = if_processed
        self.concat_mode = concat_mode
        self.organism = organism

        # Store individual omics data in a dictionary for easy access
        self.omics_data_types = [
            "gene_mutation",
            "gene_expression",
            "gene_fusion",
            "gene_cnv",
            "gene_methylation",
            "mirna_expression",
            "circrna_expression",
            "protein_expression",
            "protein_phosphorylation",
            "protein_acetylation",
            "time_series_data",
            "meta_data",
            "scrna_data",
            "metabolomics_data",
            "other_data",
        ]
        self.omics_data_list = [
            self.gene_mutation,
            self.gene_expression,
            self.gene_fusion,
            self.gene_cnv,
            self.gene_methylation,
            self.mirna_expression,
            self.circrna_expression,
            self.protein_expression,
            self.protein_phosphorylation,
            self.protein_acetylation,
            self.time_series_data,
            self.meta_data,
            self.scrna_data,
            self.metabolomics_data,
            self.other_data,
        ]
        self.multi_omics_data = dict(
            (data_type, omics_data)
            for data_type, omics_data in zip(
                self.omics_data_types, self.omics_data_list
            )
            if omics_data is not None
        )

        # Check features and labels validity
        self._check_feature_validity()
        self.sample_names = self._concat_samples(mode=self.concat_mode)
        self.feature_names = self._concat_features()
        self.sample_mapping = self._construct_mapping()

        self.labels = self._check_label_validity(labels)
        self.label_type = self._check_label_type()

    def _concat_samples(self, mode: str) -> pd.Index:
        """

        Concatenates the indexes of multi omics data into a single index, and returns the concatenated sample index.
        Parameters:
            mode (str): The type of join to be used when combining features. Default is "inner".

        Returns:
            pd.Index: Index after concatenation.
        """

        # Get index of each omic data
        sample_name_list = [
            omics_data.get_sample_names().to_list()
            for omics_data in self.multi_omics_data.values()
        ]

        # Perform concatenation based on the mode
        if mode == "inner":
            combined_sample_names = sample_name_list[0]
            for next_sample_names in sample_name_list[1:]:
                combined_sample_names = list(
                    filter(lambda x: x in combined_sample_names, next_sample_names)
                )
        elif mode == "outer":
            combined_sample_names = sample_name_list[0]
            for next_sample_names in sample_name_list[1:]:
                combined_sample_names = list(
                    OrderedDict.fromkeys(combined_sample_names + next_sample_names)
                )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if len(combined_sample_names) == 0:
            raise ValueError("The combined index of multi omics data is empty !")

        return pd.Index(combined_sample_names)

    def _concat_features(self) -> pd.Index:
        """
        Concatenates features of multi omics data, and returns the concatenated feature names

        Returns:
            pd.Index: Index after concatenation.
        """
        feature_name_list = []
        for data_type in sorted(list(self.multi_omics_data.keys())):
            omics_data = self.multi_omics_data[data_type]
            feature_name = (
                pd.Series(omics_data.get_feature_names())
                .apply(lambda x: str(data_type) + "|" + str(x))
                .tolist()
            )
            feature_name_list.extend(feature_name)
        return pd.Index(feature_name_list)

    def _construct_mapping(self) -> pd.DataFrame:
        """
        Constructing the mapping between the concatenated sample index of multi-omics data and the sample index of single omics data
        """
        mapping = pd.DataFrame(
            index=self.sample_names, columns=list(self.multi_omics_data.keys())
        )
        for index in self.sample_names:
            for data_type, omics_data in self.multi_omics_data.items():
                single_omics_samples = omics_data.get_sample_names()
                if index in single_omics_samples:
                    mapping.loc[index, data_type] = index
        return mapping

    def get_shape(self) -> tuple:
        """
        Get the shape of the multi omics data.

        Returns:
            tuple: Shape of the multi omics data.
        """
        return (len(self.sample_names), len(self.feature_names))

    def get_labels(self) -> pd.Series:
        """
        Get the labels of data.

        Returns:
            pd.Series: The labels of the dataset.
        """
        return self.labels.loc[self.sample_names]

    def get_feature_names(self) -> pd.Index:
        """
        Get the feature names.

        Returns:
            pd.Index: The feature names.
        """
        return self.feature_names

    def get_sample_names(self) -> pd.Index:
        """
        Get the sample names.

        Returns:
            pd.Index: The sample names.
        """
        return self.sample_names

    def update_features(
        self,
        gene_mutation: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
        gene_expression: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
        gene_fusion: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
        gene_cnv: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
        gene_methylation: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
        mirna_expression: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
        circrna_expression: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
        protein_expression: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
        protein_phosphorylation: Optional[
            Union[pd.DataFrame, List[pd.DataFrame]]
        ] = None,
        protein_acetylation: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
        time_series_data: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
        meta_data: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
        scrna_data: Optional[pd.DataFrame, List[pd.DataFrame]] = None,
        metabolomics_data: Optional[pd.DataFrame, List[pd.DataFrame]] = None,
        other_data: Optional[pd.DataFrame, List[pd.DataFrame]] = None,
        inplace: bool = True,
    ) -> MultiOmicsData:
        """
        Update the specified layer with the given features

        Parameters:
            gene_mutation (pd.DataFrame, optional): Gene mutation data.
            gene_expression (pd.DataFrame, optional): Gene expression data.
            gene_fusion (pd.DataFrame, optional): Gene fusion data.
            gene_cnv (pd.DataFrame, optional): Gene CNV data.
            gene_methylation (pd.DataFrame, optional): Gene methylation data.
            mirna_expression (pd.DataFrame, optional): miRNA expression data.
            circrna_expression (pd.DataFrame, optional): circRNA expression data.
            protein_expression (pd.DataFrame, optional): Protein expression data.
            protein_phosphorylation (pd.DataFrame, optional): Protein phosphorylation data.
            protein_acetylation (pd.DataFrame, optional): Protein acetylation data.
            time_series_data (pd.DataFrame, optional): Time-series data.
            meta_data (pd.DataFrame, optional): Metadata.
            scrna_data (pd.DataFrame, optional): Single-cell RNA data.
            metabolomics_data (pd.DataFrame, optional): Metabolomics data.
            other_data (pd.DataFrame, optional): Other data.
            labels (pd.Series, optional): Labels for the dataset.
            organism (str): Organism type.
            if_processed (bool): Flag to indicate if data is processed.
            concat_mode(str): The mode of concat features, must be one of ["inner", "outer"].
            inplace (bool, optional): If True, the `MultiOmicsData` object is updated in-place. If False, a new object is returned.

        Returns:
            DataBase: A new or updated MultiOmicsData object
        """
        if inplace:
            self.__init__(
                gene_mutation=gene_mutation,
                gene_expression=gene_expression,
                gene_fusion=gene_fusion,
                gene_cnv=gene_cnv,
                gene_methylation=gene_methylation,
                mirna_expression=mirna_expression,
                circrna_expression=circrna_expression,
                protein_expression=protein_expression,
                protein_phosphorylation=protein_phosphorylation,
                protein_acetylation=protein_acetylation,
                time_series_data=time_series_data,
                meta_data=meta_data,
                scrna_data=scrna_data,
                metabolomics_data=metabolomics_data,
                other_data=other_data,
                labels=self.labels,
                organism=self.organism,
                if_processed=self.if_processed,
                concat_mode=self.concat_mode,
            )
            return self
        else:
            return self.__class__(
                gene_mutation=gene_mutation,
                gene_expression=gene_expression,
                gene_fusion=gene_fusion,
                gene_cnv=gene_cnv,
                gene_methylation=gene_methylation,
                mirna_expression=mirna_expression,
                circrna_expression=circrna_expression,
                protein_expression=protein_expression,
                protein_phosphorylation=protein_phosphorylation,
                protein_acetylation=protein_acetylation,
                time_series_data=time_series_data,
                meta_data=meta_data,
                scrna_data=scrna_data,
                metabolomics_data=metabolomics_data,
                other_data=other_data,
                labels=self.labels,
                organism=self.organism,
                if_processed=self.if_processed,
                concat_mode=self.concat_mode,
            )

    def update_labels(
        self, labels: Union[pd.Series, List[pd.Series]], inplace: bool = True
    ) -> MultiOmicsData:
        """
        Update the specified layer with the given features

        Parameters:
            labels (pd.Series): The label of the data.
            inplace (bool, optional): If True, the `DataBase` object is updated in-place. If False, a new object is returned.
        Returns:
            DataBase: A new or updated MultiOmicsData object
        """
        if inplace:
            self.__init__(
                gene_mutation=self.get_single_omics_features("gene_mutation"),
                gene_expression=self.get_single_omics_features("gene_expression"),
                gene_fusion=self.get_single_omics_features("gene_fusion"),
                gene_cnv=self.get_single_omics_features("gene_cnv"),
                gene_methylation=self.get_single_omics_features("gene_methylation"),
                mirna_expression=self.get_single_omics_features("mirna_expression"),
                circrna_expression=self.get_single_omics_features("circrna_expression"),
                protein_expression=self.get_single_omics_features("protein_expression"),
                protein_phosphorylation=self.get_single_omics_features(
                    "protein_phosphorylation"
                ),
                protein_acetylation=self.get_single_omics_features(
                    "protein_acetylation"
                ),
                time_series_data=self.get_single_omics_features("time_series_data"),
                meta_data=self.get_single_omics_features("meta_data"),
                scrna_data=self.get_single_omics_features("scrna_data"),
                metabolomics_data=self.get_single_omics_features("metabolomics_data"),
                other_data=self.get_single_omics_features("other_data"),
                labels=labels,
                organism=self.organism,
                if_processed=self.if_processed,
                concat_mode=self.concat_mode,
            )
            return self
        else:
            return self.__class__(
                gene_mutation=self.get_single_omics_features("gene_mutation"),
                gene_expression=self.get_single_omics_features("gene_expression"),
                gene_fusion=self.get_single_omics_features("gene_fusion"),
                gene_cnv=self.get_single_omics_features("gene_cnv"),
                gene_methylation=self.get_single_omics_features("gene_methylation"),
                mirna_expression=self.get_single_omics_features("mirna_expression"),
                circrna_expression=self.get_single_omics_features("circrna_expression"),
                protein_expression=self.get_single_omics_features("protein_expression"),
                protein_phosphorylation=self.get_single_omics_features(
                    "protein_phosphorylation"
                ),
                protein_acetylation=self.get_single_omics_features(
                    "protein_acetylation"
                ),
                time_series_data=self.get_single_omics_features("time_series_data"),
                meta_data=self.get_single_omics_features("meta_data"),
                scrna_data=self.get_single_omics_features("scrna_data"),
                metabolomics_data=self.get_single_omics_features("metabolomics_data"),
                other_data=self.get_single_omics_features("other_data"),
                labels=labels,
                organism=self.organism,
                if_processed=self.if_processed,
                concat_mode=self.concat_mode,
            )

    def get_single_omics_features(self, omics_data_type: str) -> pd.DataFrame:
        """
        Get features of single omics data

        Parameters:
            omics_data_type (str): The type of omics data, must be in:
                - "gene_mutation",
                - "gene_expression",
                - "gene_fusion",
                - "gene_cnv",
                - "gene_methylation",
                - "mirna_expression",
                - "circrna_expression",
                - "protein_expression",
                - "protein_phosphorylation",
                - "protein_acetylation",
                - "time_series_data",
                - "meta_data",
                - "scrna_data",
                - "metabolomics_data",
                - "other_data"
        Returns:
            pd.DataFrame: The feature matrix of target omics data
        """
        if omics_data_type not in self.omics_data_types:
            raise ValueError("Error omics data type {}".format(omics_data_type))

        if omics_data_type not in self.multi_omics_data:
            return None

        layer = "processed" if self.if_processed else "raw"
        return self.multi_omics_data[omics_data_type].get_features(layer)

    def update_processed_flag(self, inplace: bool = True) -> None:
        """
        After all preprocessing is completed, if_processed is set to True

        Parameters:
            inplace (bool, optional): If True, the `DataBase` object is updated in-place. If False, a new object is returned.
        Returns:
            DataBase: A new or updated DataBase object
        """
        process_flags = [
            omics_data.if_processed for omics_data in self.multi_omics_data.values()
        ]
        if_processed = True if all(process_flags) else False
        if inplace:
            self.if_processed = if_processed
            self.sample_names = self._concat_samples(mode=self.concat_mode)
            self.feature_names = self._concat_features()
            return self
        else:
            return self.__class__(
                gene_mutation=self.get_single_omics_features("gene_mutation"),
                gene_expression=self.get_single_omics_features("gene_expression"),
                gene_fusion=self.get_single_omics_features("gene_fusion"),
                gene_cnv=self.get_single_omics_features("gene_cnv"),
                gene_methylation=self.get_single_omics_features("gene_methylation"),
                mirna_expression=self.get_single_omics_features("mirna_expression"),
                circrna_expression=self.get_single_omics_features("circrna_expression"),
                protein_expression=self.get_single_omics_features("protein_expression"),
                protein_phosphorylation=self.get_single_omics_features(
                    "protein_phosphorylation"
                ),
                protein_acetylation=self.get_single_omics_features(
                    "protein_acetylation"
                ),
                time_series_data=self.get_single_omics_features("time_series_data"),
                meta_data=self.get_single_omics_features("meta_data"),
                scrna_data=self.get_single_omics_features("scrna_data"),
                metabolomics_data=self.get_single_omics_features("metabolomics_data"),
                other_data=self.get_single_omics_features("other_data"),
                labels=self.labels,
                organism=self.organism,
                if_processed=if_processed,
                concat_mode=self.concat_mode,
            )

    def _check_feature_validity(self) -> tuple:
        """
        Checks the validity of the features by ensuring not empty.
        """
        if len(self.multi_omics_data) == 0:
            raise ValueError("Empty multi_omics_data!")

    def _check_label_validity(self, labels: Union[pd.Series, list[pd.Series]]) -> tuple:
        """
        Checks the validity of the labels by ensuring not empty.
        """
        assert (
            isinstance(labels, pd.Series) or labels is None or isinstance(labels, list)
        ), "Labels shold be pd.Series type or list of pd.Series!"
        if isinstance(labels, list):
            assert len(labels) > 0, "Labels list is empty!"
            for label in labels:
                assert isinstance(
                    label, pd.Series
                ), "Each item in label list should be Series type"
            labels = pd.concat(labels, axis=0, join="outer")

        return labels

    def __getitem__(self, index) -> MultiOmicsData:
        """
        Slice or index the data and return a new DataBase instance with the sliced data.

        Parameters:
            index (slice, list, tuple, or array-like): The index or slice to apply. Supports both row and column indexing.

        Returns:
            DataBase: A new DataBase object containing the sliced data.
        """
        if isinstance(index, tuple):
            row_idx, col_idx = index
        else:
            row_idx, col_idx = index, slice(None)

        # Convert the target index to the original index in each omics
        if isinstance(row_idx, int):
            original_index = self.sample_mapping.iloc[row_idx]
        elif isinstance(row_idx, slice):
            original_index = self.sample_mapping[row_idx]
        else:
            if row_idx[0] not in self.sample_names:
                original_index = self.sample_mapping.iloc[row_idx]
            else:
                original_index = self.sample_mapping[row_idx]

        sliced_omics_data = {}
        for data_type, omics_data in self.multi_omics_data.items():
            omics_index = original_index[data_type]
            sliced_omics_data[data_type] = omics_data[omics_index]

        layer = "processed" if self.if_processed else "raw"
        return self.__class__(
            gene_mutation=(
                sliced_omics_data["gene_mutation"].get_features(layer)
                if "gene_mutation" in sliced_omics_data
                else None
            ),
            gene_expression=(
                sliced_omics_data["gene_expression"].get_features(layer)
                if "gene_expression" in sliced_omics_data
                else None
            ),
            gene_fusion=(
                sliced_omics_data["gene_fusion"].get_features(layer)
                if "gene_fusion" in sliced_omics_data
                else None
            ),
            gene_cnv=(
                sliced_omics_data["gene_cnv"].get_features(layer)
                if "gene_cnv" in sliced_omics_data
                else None
            ),
            gene_methylation=(
                sliced_omics_data["gene_methylation"].get_features(layer)
                if "gene_methylation" in sliced_omics_data
                else None
            ),
            mirna_expression=(
                sliced_omics_data["mirna_expression"].get_features(layer)
                if "mirna_expression" in sliced_omics_data
                else None
            ),
            circrna_expression=(
                sliced_omics_data["circrna_expression"].get_features(layer)
                if "circrna_expression" in sliced_omics_data
                else None
            ),
            protein_expression=(
                sliced_omics_data["protein_expression"].get_features(layer)
                if "protein_expression" in sliced_omics_data
                else None
            ),
            protein_phosphorylation=(
                sliced_omics_data["protein_phosphorylation"].get_features(layer)
                if "protein_phosphorylation" in sliced_omics_data
                else None
            ),
            protein_acetylation=(
                sliced_omics_data["protein_acetylation"].get_features(layer)
                if "protein_acetylation" in sliced_omics_data
                else None
            ),
            time_series_data=(
                sliced_omics_data["time_series_data"].get_features(layer)
                if "time_series_data" in sliced_omics_data
                else None
            ),
            meta_data=(
                sliced_omics_data["meta_data"].get_features(layer)
                if "meta_data" in sliced_omics_data
                else None
            ),
            scrna_data=(
                sliced_omics_data["scrna_data"].get_features(layer)
                if "scrna_data" in sliced_omics_data
                else None
            ),
            metabolomics_data=(
                sliced_omics_data["metabolomics_data"].get_features(layer)
                if "metabolomics_data" in sliced_omics_data
                else None
            ),
            other_data=(
                sliced_omics_data["other_data"].get_features(layer)
                if "other_data" in sliced_omics_data
                else None
            ),
            labels=self.labels,
            organism=self.organism,
            if_processed=self.if_processed,
            concat_mode=self.concat_mode,
        )

    def __len__(self) -> int:
        """
        Get the length of the data.

        Returns:
            int: The length of the data.
        """
        return self.get_shape()[0]

    def get_features(
        self, layer="raw", selected_features=None
    ) -> pd.DataFrame:
        """
        Concatenate features from different omics data into a single DataFrame.

        Parameters:
            mode (str): The type of join to be used when combining features. Default is "inner".
            layer (str): The layer to retrieve features from. Must be 'raw' or 'processed'.
            selected_features(list, or pd.series): The index of selected features names.

        Returns:
            pd.DataFrame: The concatenated feature matrix.
        """
        sliced_data = self[:]
        feature_dfs = []
        for data_type in sorted(list(self.multi_omics_data.keys())):
            features = sliced_data.multi_omics_data[data_type].get_features(layer=layer).values
            feature_dfs.append(features)


        combined_features = np.concatenate(feature_dfs, axis=1)
        combined_features = pd.DataFrame(
            combined_features, index=self.sample_names, columns=self.feature_names
        )
        if selected_features is None:
            return combined_features

        if combined_features is None:
            raise ValueError(
                "processed features is None, can not get selected_features"
            )
        return pd.DataFrame(combined_features[selected_features])

    def get_raw_features(self, selected_features=None) -> pd.DataFrame:
        """
        Concatenate all omic data features in the given mode.

        Parameters:
            mode (str): The type of join to be used when combining features. Default is "inner".

        Returns:
            pd.DataFrame: The concatenated raw feature matrix.
        """
        raw_features = []
        new_feature_names = []
        for data_type in sorted(list(self.multi_omics_data.keys())):
            raw_feature = self.multi_omics_data[data_type].get_features(layer="raw")
            feature_names = [data_type+"|"+column for column in raw_feature.columns]
            new_feature_names.extend(feature_names)
            raw_features.append(raw_feature)
        
        if self.concat_mode == "inner":
            combined_raw_features = pd.concat(raw_features, axis=1, join="inner")
        elif self.concat_mode == "outer":
            combined_raw_features = pd.concat(raw_features, axis=1, join="outer")
        else:
            raise ValueError(f"Unsupported mode: {self.concat_mode}")

        combined_raw_features.columns = new_feature_names
        
        if selected_features is None:
            return combined_raw_features
        
        return pd.DataFrame(combined_raw_features[selected_features])
    
    def copy(self) -> MultiOmicsData:
        """
        Copy of MultiOmicsData object

        Returns:
            MultiOmicsData: A new object.
        """
        return self.__class__(
            gene_mutation=self.get_single_omics_features("gene_mutation"),
            gene_expression=self.get_single_omics_features("gene_expression"),
            gene_fusion=self.get_single_omics_features("gene_fusion"),
            gene_cnv=self.get_single_omics_features("gene_cnv"),
            gene_methylation=self.get_single_omics_features("gene_methylation"),
            mirna_expression=self.get_single_omics_features("mirna_expression"),
            circrna_expression=self.get_single_omics_features("circrna_expression"),
            protein_expression=self.get_single_omics_features("protein_expression"),
            protein_phosphorylation=self.get_single_omics_features(
                "protein_phosphorylation"
            ),
            protein_acetylation=self.get_single_omics_features("protein_acetylation"),
            time_series_data=self.get_single_omics_features("time_series_data"),
            meta_data=self.get_single_omics_features("meta_data"),
            scrna_data=self.get_single_omics_features("scrna_data"),
            metabolomics_data=self.get_single_omics_features("metabolomics_data"),
            other_data=self.get_single_omics_features("other_data"),
            labels=self.labels,
            organism=self.organism,
            if_processed=self.if_processed,
            concat_mode=self.concat_mode,
        )

    def _check_label_type(self) -> str:
        """
        check the type of label, and assign to self.label_type, can be "classification" or "regression"
        """
        if self.labels is None:
            return None
        num_unique_values = self.labels.nunique()
        if num_unique_values > 2:
            return "regression"
        else:
            return "classification"

    def get_label_type(self) -> pd.Series:
        """
        Get the type of label.

        Returns:
            string: The label type.
        """
        return self.label_type


if __name__ == "__main__":
    mut = pd.read_csv("simu_data/mutation.csv", index_col=0)
    expr = pd.read_csv("simu_data/expression.csv", index_col=0)
    prot = pd.read_csv("simu_data/protein.csv", index_col=0)
    cnv = pd.read_csv("simu_data/cnv.csv", index_col=0)
    label = pd.read_csv("simu_data/label.csv", index_col=0)

    data = MultiOmicsData(
        gene_mutation=mut,
        gene_expression=expr,
        protein_expression=prot,
        gene_cnv=cnv,
        labels=label["label"],
    )