import sys
import os
import json
scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(scripts_dir, "mlgenie"))
import MLGenie.Data as MLGenie_Data
import pickle
import pandas as pd
import argparse
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
class MLGeniePredictor:
    def __init__(self, test_data:dict, model_path:str=MODEL_PATH, threshold:float=0.46,control_group_labels:list=["0"],case_group_labels:list=["1"],required_cols:list=["SampleID", "GAPDH", "RB012", "RB018", "RB020", "RB054", "RB080", "RB102", "RB117", "RB167"]):
        self.model_path = model_path
        self.test_data = test_data
        self.threshold = threshold
        self.ctrl_group_labels = control_group_labels
        self.case_group_labels = case_group_labels
        self.analyzer = pickle.load(open(self.model_path, "rb"))
        self.required_cols = required_cols
       
    def _validate_data(self, test_df:pd.DataFrame):
        """
        Validate the input data,the input data should contain the required columns
        :param test_df: Input DataFrame, index is sample_id
        :return: None
        """
        missing_cols = [col for col in self.required_cols if col not in test_df.columns]
        if missing_cols:
            missing_cols_str = ", ".join(missing_cols)
            raise ValueError(f"The input data is missing the following columns: {missing_cols_str}")

    def _check_duplicate_sample_ids(self, test_df:pd.DataFrame):
        """
        Check whether the sample ids are duplicated
        :param test_df: Input DataFrame
        :return: None
        """
        # Check for duplicate sample IDs
        duplicate_ids = test_df["SampleID"].duplicated()
        if duplicate_ids.any():
            duplicated_sample_ids = test_df.loc[duplicate_ids, "SampleID"].tolist()
            duplicated_str = ", ".join(map(str, duplicated_sample_ids))
            raise ValueError(f"Duplicate sample IDs found: {duplicated_str}")
       

        
    def load_test_data(self):
        """
        Load test data
        :return: DataFrame
        """
        test_df = pd.DataFrame(self.test_data)
        if "SampleID" not in test_df.columns:
            raise ValueError("The input data is missing the SampleID column")
        else:
            test_df["SampleID"] = test_df["SampleID"].astype(str)
        return test_df

    def check_samples(self, test_df:pd.DataFrame):
        """
        Check incomplete and invalid data
        :param test_df: Input DataFrame, index is sample_id
        :return: valid_df, invalid_list, incomplete_list
        """
        test_df = test_df[self.required_cols]
        test_df = test_df.set_index("SampleID")
        test_df.index.name = "sample_id"

        gene_cols = ["GAPDH", "RB012", "RB018", "RB020", "RB054", "RB080", "RB102", "RB117", "RB167"]
        # 1. First, check incomplete data
        incomplete_mask = []
        for idx, row in test_df.iterrows():
            incomplete = False
            for col in gene_cols:
                val = row[col]
                if val == 'Undetermined':
                    continue
                if pd.isna(val):
                    incomplete = True
                    break
                try:
                    float(val)
                except Exception:
                    incomplete = True
                    break
            incomplete_mask.append(incomplete)
        incomplete_df = test_df[incomplete_mask]
        # 2. Check invalid data in the remaining samples
        complete_df = test_df[[not x for x in incomplete_mask]]
        # GAPDH >= 30 or any column in gene_cols > 40
        gapdh_numeric = pd.to_numeric(complete_df["GAPDH"], errors="coerce")
        # Check if any column in gene_cols > 40
        gene_numeric = complete_df[gene_cols].apply(pd.to_numeric, errors="coerce")
        over_40_mask = (gene_numeric > 40).any(axis=1)
        invalid_mask = (gapdh_numeric >= 30) | over_40_mask
        invalid_df = complete_df[invalid_mask]
        valid_df = complete_df[~invalid_mask]
        #valid_df["sample_id"] = valid_df.index
        # Only keep sample ID (index) and convert to list
        incomplete_list = list(incomplete_df.index)
        invalid_list = list(invalid_df.index)
        return valid_df, invalid_list, incomplete_list

    def transform_data(self, vaild_df:pd.DataFrame):
        """
        Transform the input data,replace 'Undetermined' with 40
        :param vaild_df: Input DataFrame, index is sample_id
        :return: DataFrame
        """
        cols_to_replace = [col for col in vaild_df.columns if col != 'SampleID']
        vaild_df[cols_to_replace] = vaild_df[cols_to_replace].replace('Undetermined', 40)
        return vaild_df

    def get_label_df(self, test_df: pd.DataFrame, seed: int = 42):
        """
        Generate labels for the input DataFrame based on the number of samples:
        - If there is only one sample, the label is 1
        - If there are multiple samples, the first sample is 0 and the rest are 1
        """
        n = len(test_df)
        if n == 1:
            labels = [1]
        else:
            labels = [0] + [1] * (n - 1)
        label_df = pd.DataFrame(labels, index=test_df.index, columns=["label"])
        label_df.index.name = "sample_id"
        return label_df

    def predict_proba(self,vaild_df:pd.DataFrame,invalid_list:list,incomplete_list:list):
        """
        Predict the probability of the input DataFrame
        :param vaild_df: Input DataFrame, index is sample_id
        :param invalid_list: List of invalid samples
        :param incomplete_list: List of incomplete samples
        :return: DataFrame
        """
        labels = self.get_label_df(vaild_df)
        data = MLGenie_Data.MultiOmicsData(gene_expression=[vaild_df], labels=labels["label"])
        test_pred, test_performance, test_ROC_data, test_PR_data = self.analyzer.transform(data)
        test_pred["sample_id"] = vaild_df.index
        test_pred = test_pred[["sample_id", "prediction"]]
        test_pred["result"] = test_pred["prediction"].apply(lambda x: "High risk" if x >= self.threshold else "Low risk")
        test_pred.columns = ["SampleID", "Predict Score", "Result"]
        
        # Deal with invalid samples
        for sample_id in invalid_list:
            invalid_row = pd.DataFrame({
                "SampleID": [sample_id],
                "Predict Score": [""],
                "Result": ["INVALID DATA"]
            })
            test_pred = pd.concat([test_pred, invalid_row], ignore_index=True)
        # Deal with incomplete samples
        for sample_id in incomplete_list:
            incomplete_row = pd.DataFrame({
                "SampleID": [sample_id],
                "Predict Score": [""],
                "Result": ["INCOMPLETE DATA"]
            })
            test_pred = pd.concat([test_pred, incomplete_row], ignore_index=True)
        result_json_str = test_pred.to_json(orient="records", force_ascii=False)
        result = json.loads(result_json_str)
        return result

    def run(self):
        test_df = self.load_test_data()
        self._validate_data(test_df)
        self._check_duplicate_sample_ids(test_df)
        valid_df, invalid_list, incomplete_list = self.check_samples(test_df)
        if len(invalid_list) > 0:
            invalid_list_str = ", ".join(invalid_list)
            print(f"Invalid samples: {invalid_list_str}")
        if len(incomplete_list) > 0:
            incomplete_list_str = ", ".join(incomplete_list)
            print(f"Incomplete samples: {incomplete_list_str}")
        transform_valid_df = self.transform_data(valid_df)
        return self.predict_proba(transform_valid_df,invalid_list,incomplete_list)
    
    


