import pandas as pd
import numpy as np
import shap


class gene_name_mapper:
    """
    Read the gene name mapper to map gene names to different naming systems
    """

    def __init__(self):
        """
        Initialize the gene name mapper by reading the HGNC dataset
        """
        self.gene_names = pd.read_csv(
            "hgnc_complete_set.txt",
            sep="\t",
            dtype=str,
            usecols=[
                "hgnc_id",
                "symbol",
                "entrez_id",
                "ensembl_gene_id",
                "refseq_accession",
            ],
        )
        self.gene_names.columns = ["hgnc", "symbol", "entrez", "ensembl", "refseq"]

    def convert(self, genes, in_name="auto", out_name="symbol"):
        """
        Convert gene names based on gene name table

        Parameters:
        -----------
        genes : list or pandas.Series
            List of gene names to convert
        in_name : str, optional
            Input naming system (default is "auto")
        out_name : str, optional
            Output naming system (default is "symbol")

        Returns:
        --------
        pandas.Series
            Converted gene names
        """
        valid_names = ["hgnc", "symbol", "entrez", "ensembl", "refseq"]

        if in_name != "auto" and in_name not in valid_names:
            raise ValueError(f"in_name must be 'auto' or one of {valid_names}")

        if out_name not in valid_names:
            raise ValueError(f"out_name must be one of {valid_names}")

        genes = pd.Series(genes).astype(str)

        if in_name == "auto":
            overlap = pd.Series(
                {name: genes.isin(self.gene_names[name]).mean() for name in valid_names}
            )
            max_overlap = overlap.max()
            inferred_name = overlap.idxmax()

            if max_overlap > 0.5:
                in_name = inferred_name
                print(
                    f"Inferred input naming system: {in_name} (overlap: {max_overlap:.2%})"
                )
            else:
                raise ValueError(
                    f"Could not automatically determine input naming system. Max overlap was only {max_overlap:.2%}"
                )

        # Create a mapping dictionary from input naming system to output naming system
        name_map = dict(zip(self.gene_names[in_name], self.gene_names[out_name]))

        # Convert gene names, using the original name if it can't be mapped
        converted_genes = genes.map(lambda x: name_map.get(x, x))

        # Count and report unmapped genes
        unmapped_count = (converted_genes == genes).sum()
        if unmapped_count > 0:
            print(
                f"Warning: {unmapped_count} genes could not be mapped and retain their original names."
            )

        return converted_genes

def compute_feature_shap_values(model, X, task):
    """
    Calculate SHAP values for each feature, and store the raw data used to generate the plot.

    Parameters:
    model: The machine learning model for which to calculate SHAP values.
    X: The input data for which to calculate SHAP values.
    task: The type of task, either "classification" or "regression".

    Returns:
    shap_obj: The calculated SHAP object containing SHAP values.
    shap_data: The raw SHAP values data combined with scaled feature values.
    """
    if task not in ["classification", "regression"]:
        raise ValueError("task should be 'classification' or 'regression'")
    
    # Create a SHAP explainer
    explainer = shap.Explainer(model.predict, X)
    
    # Calculate SHAP values
    shap_obj = explainer(X)

    # if task == "classification":
    #     shap_obj = shap_obj[..., 1]
    
    # Store the raw SHAP values data
    shap_values = shap_obj.values
    
    # Prepare SHAP values for further analysis
    shap_values = pd.DataFrame(shap_values, columns=X.columns, index = X.index)
    mean_abs_shap = shap_values.abs().mean(axis=0).sort_values(ascending=False)
    shap_values = shap_values.loc[:,mean_abs_shap.index]
    shap_values = shap_values.reset_index().melt(id_vars="index", var_name="feature", value_name="shap_value")

    # Scale each feature in X to [0,1]
    X_scale = (X - X.min()) / (X.max() - X.min())
    X_scale = X_scale.reset_index().melt(id_vars="index", var_name="feature", value_name="feature_scale")

    # raw X values
    X_values = X.reset_index().melt(id_vars="index", var_name="feature", value_name="feature_value")

    shap_data = pd.merge(shap_values, X_scale, on=["index", "feature"])
    shap_data = pd.merge(shap_data, X_values, on=["index", "feature"]).rename(columns={"index": "sample"})
    
    return shap_obj, shap_data

def check_X_y(X, y):
    """
    Check the input data and labels.

    Parameters:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The labels of the dataset.

    Returns:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The labels of the dataset.
    """
    if X is None or not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if y is None or not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("The number of samples in X and y must be the same.")

if __name__ == "__main__":
    
    shap_obj, shap_data = compute_feature_shap_values(model, X, task)

    # use shap to plot SHAP dot plot for all features
    shap.summary_plot(shap_obj, X, plot_type="dot")

    # plot SHAP plot using SHAP values
    sns.stripplot(data=shap_data, x="shap_value", y="feature", hue="feature_scale")

    # use shap to plot SHAP dependence plots
    shap.plots.scatter(shap_obj[:, feat_name], show=False)

    # plot SHAP dependence plots using SHAP values
    sns.scatterplot(data=shap_data.loc[shap_data["feature"] == feat_name,:], x="feature_value", y="shap_value")
