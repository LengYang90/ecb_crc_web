import numpy as np
import pandas as pd
from scipy.stats import norm


def simulate_data(
    n_samples, label,
    n_feat_normal, n_feat_normal_bimodal, 
    n_feat_count, n_feat_count_bimodal,
    n_feat_binary, n_feat_uniform, 
    n_base, n_interact, n_correlated, 
    corr_level, noise_level_X, noise_level_y
):
    """
    Simulates a dataset with various types of features and generates labels based on the specified type.

    Parameters:
    - n_samples (int): Number of samples to generate.
    - label (str): Type of labels to generate. Options are 'regression' or 'classification'.
    - n_feat_normal (int): Number of normal features.
    - n_feat_normal_bimodal (int): Number of normal bimodal features.
    - n_feat_count (int): Number of count features.
    - n_feat_count_bimodal (int): Number of count bimodal features.
    - n_feat_binary (int): Number of binary features.
    - n_feat_uniform (int): Number of uniform features.
    - n_base (int): Number of features to be used as base features.
    - n_interact (int): Number of features of interaction terms.
    - n_correlated (int): Number of features to be correlated with the base features.
    - corr_level (str): Correlation level for features. Options are 'low', 'medium', 'high'.
    - noise_level_X (str): Noise level for features. Options are 'low', 'medium', 'high'.
    - noise_level_y (str): Noise level for labels. Options are 'low', 'medium', 'high'.

    Returns:
    - tuple: (features, y, coef)
      - features (pd.DataFrame): Combined feature matrix of shape (n_samples, n_features).
      - y (np.ndarray): Label vector of shape (n_samples,).
      - coef (pd.DataFrame): DataFrame of coefficients used for generating labels.
    """

    # Define noise and correlation parameters
    rhos = {"low": 0.4, "medium": 0.6, "high": 0.8}
    noise_normal = {"low": 0.1, "medium": 0.3, "high": 0.5}
    noise_count = {"low": 1.2, "medium": 1.5, "high": 2}
    noise_binary = {"low": 0.05, "medium": 0.1, "high": 0.2}
    noise_uniform = {"low": 0.05, "medium": 0.1, "high": 0.2}
    noise_label = {"low": 0.05, "medium": 0.1, "high": 0.2}
    
    # Get the correlation and noise levels
    rho = rhos[corr_level]
    sigma = noise_normal[noise_level_X]
    c = noise_count[noise_level_X]
    flip_rate = noise_binary[noise_level_X]
    epsilon = noise_uniform[noise_level_X]
    noise_y = noise_label[noise_level_y]

    # ============== Generate features ============ #
    features = pd.DataFrame()

    # -------- 1. normal features --------- #
    if n_feat_normal > 0:
        X_normal, X_normal_corr, X_normal_noise = simulate_base_matrix(
            n_samples, n_feat_normal, n_base, n_correlated, rho
        )
        # concat features
        X_normal_combined = np.concatenate([X_normal, X_normal_corr, X_normal_noise], axis=1)
        # simulate normal data
        feat_normal = simulate_normal_data(X_normal_combined, sigma)
        # add feature name
        feat_normal = pd.DataFrame(
            feat_normal, columns=[f"feat_normal_{i+1}" for i in range(feat_normal.shape[1])]
        )
        features = pd.concat([features, feat_normal], axis=1)

    # -------- 2. normal bimodal features --------- #
    if n_feat_normal_bimodal > 0:
        X_normal_bimodal, X_normal_corr_bimodal, X_normal_noise_bimodal = simulate_base_matrix(
            n_samples, n_feat_normal_bimodal, n_base, n_correlated, rho, bimodal=True
        )
        # concat features
        X_normal_bimodal_combined = np.concatenate([X_normal_bimodal, X_normal_corr_bimodal, X_normal_noise_bimodal], axis=1)
        # simulate bimodal normal data
        feat_normal_bimodal = simulate_normal_data(X_normal_bimodal_combined, sigma)
        # add feature name
        feat_normal_bimodal = pd.DataFrame(
            feat_normal_bimodal, columns=[f"feat_normal_bimodal_{i+1}" for i in range(feat_normal_bimodal.shape[1])]
        )
        features = pd.concat([features, feat_normal_bimodal], axis=1)

    # -------- 3. count features --------- #
    if n_feat_count > 0:
        X_count, X_count_corr, X_count_noise = simulate_base_matrix(n_samples, n_feat_count, n_base, n_correlated, rho)
        # concat features
        X_count_combined = np.concatenate([X_count, X_count_corr, X_count_noise], axis=1)
        # simulate count data
        feat_count = simulate_count_data_negative_binomial(X_count_combined, c)
        # add feature name
        feat_count = pd.DataFrame(
            feat_count, columns=[f"feat_count_{i+1}" for i in range(feat_count.shape[1])]
        )
        features = pd.concat([features, feat_count], axis=1)

    # -------- 4. count bimodal features --------- #
    if n_feat_count_bimodal > 0:
        X_count_bimodal, X_count_corr_bimodal, X_count_noise_bimodal = simulate_base_matrix(
            n_samples, n_feat_count_bimodal, n_base, n_correlated, rho, bimodal=True
        )
        # concat features
        X_count_bimodal_combined = np.concatenate([X_count_bimodal, X_count_corr_bimodal, X_count_noise_bimodal], axis=1)
        # simulate count bimodal data
        feat_count_bimodal = simulate_count_data_negative_binomial(X_count_bimodal_combined, c)
        # add feature name
        feat_count_bimodal = pd.DataFrame(
            feat_count_bimodal, columns=[f"feat_count_bimodal_{i+1}" for i in range(feat_count_bimodal.shape[1])]
        )
        features = pd.concat([features, feat_count_bimodal], axis=1)

    # -------- 5. binary features --------- #
    if n_feat_binary > 0:
        X_binary, X_binary_corr, X_binary_noise = simulate_base_matrix(
            n_samples, n_feat_binary, n_base, n_correlated, rho
        )
        # concat features
        X_binary_combined = np.concatenate([X_binary, X_binary_corr, X_binary_noise], axis=1)
        # simulate binary data
        feat_binary = simulate_binary_data(X_binary_combined, flip_rate)
        # add feature name
        feat_binary = pd.DataFrame(
            feat_binary, columns=[f"feat_binary_{i+1}" for i in range(feat_binary.shape[1])]
        )
        features = pd.concat([features, feat_binary], axis=1)

    # -------- 6. uniform features --------- #
    if n_feat_uniform > 0:
        X_uniform, X_uniform_corr, X_uniform_noise = simulate_base_matrix(
            n_samples, n_feat_uniform, n_base, n_correlated, rho
        )
        # concat features
        X_uniform_combined = np.concatenate([X_uniform, X_uniform_corr, X_uniform_noise], axis=1)
        # simulate uniform features
        feat_uniform = simulate_uniform_data(X_uniform_combined, epsilon)
        # add feature name
        feat_uniform = pd.DataFrame(
            feat_uniform, columns=[f"feat_uniform_{i+1}" for i in range(feat_uniform.shape[1])]
        )
        features = pd.concat([features, feat_uniform], axis=1)

    # ============= Generate labels ================= #
    # Combine all base X
    X_base_combined = pd.DataFrame()
    X_base_names = []

    if n_feat_normal > 0:
        X_base_combined = pd.concat([X_base_combined, pd.DataFrame(X_normal)], axis=1)
        X_base_names.extend(feat_normal.columns.tolist()[:X_normal.shape[1]])
    
    if n_feat_normal_bimodal > 0:
        X_base_combined = pd.concat([X_base_combined, pd.DataFrame(X_normal_bimodal)], axis=1)
        X_base_names.extend(feat_normal_bimodal.columns.tolist()[:X_normal_bimodal.shape[1]])
    
    if n_feat_count > 0:
        X_base_combined = pd.concat([X_base_combined, pd.DataFrame(X_count)], axis=1)
        X_base_names.extend(feat_count.columns.tolist()[:X_count.shape[1]])
    
    if n_feat_count_bimodal > 0:
        X_base_combined = pd.concat([X_base_combined, pd.DataFrame(X_count_bimodal)], axis=1)
        X_base_names.extend(feat_count_bimodal.columns.tolist()[:X_count_bimodal.shape[1]])
    
    if n_feat_binary > 0:
        X_base_combined = pd.concat([X_base_combined, pd.DataFrame(X_binary)], axis=1)
        X_base_names.extend(feat_binary.columns.tolist()[:X_binary.shape[1]])
    
    if n_feat_uniform > 0:
        X_base_combined = pd.concat([X_base_combined, pd.DataFrame(X_uniform)], axis=1)
        X_base_names.extend(feat_uniform.columns.tolist()[:X_uniform.shape[1]])

    X_base_combined.columns = X_base_names
    y, coef = generate_labels(X_base_combined, n_interact, label, noise_y)

    return features, y, coef

def simulate_base_matrix(n_samples, n_features, n_base, n_correlated, rho, bimodal=False):
    """
    Simulates a matrix with a mix of base features, correlated features, and noise features.

    Parameters:
    - n_samples (int): Number of samples in the dataset.
    - n_features (int): Total number of features in the dataset.
    - n_base (float): Number of features to be used as base features.
    - n_correlated (float): Number of features to be correlated with the base features.
    - rho (float): Maximum correlation coefficient for the correlated features.
    - bimodal (bool): Whether to generate bimodal distribution for base features.

    Returns:
    - tuple: (X_base, X_corr, X_noise)
      - X_base (np.ndarray): Base features matrix of shape (n_samples, n_base).
      - X_corr (np.ndarray): Correlated features matrix of shape (n_samples, n_correlated).
      - X_noise (np.ndarray): Noise features matrix of shape (n_samples, n_noise).
    """
    
    # Number of features in each category
    n_noise = n_features - n_base - n_correlated

    # Generate base features
    if bimodal:
        X_base = simulate_normal_bimodal_matrix(n_samples, n_base)
    else:
        X_base = np.random.normal(0, 1, (n_samples, n_base))
    
    # Initialize correlated features matrix
    X_corr = np.zeros((n_samples, n_correlated))

    # Generate correlated features
    for i in range(n_correlated):
        # Randomly select a base feature to correlate with
        base_idx = np.random.randint(0, n_base)
        
        # Generate correlation coefficient
        corr_coef = np.random.uniform(-rho, rho)
        
        # Generate noise
        noise = np.random.normal(0, 1, n_samples)
        
        # Generate correlated feature
        X_corr[:, i] = corr_coef * X_base[:, base_idx] + np.sqrt(1 - corr_coef**2) * noise
    
    # Generate noise features
    X_noise = np.random.normal(0, 1, (n_samples, n_noise))

    return X_base, X_corr, X_noise


def simulate_normal_data(X, sigma):
    """
    Adds noise, location shifts, and scaling to a given matrix to simulate normal data with varying noise and scaling.

    Parameters:
    - X (np.ndarray): Input data matrix of shape (n_samples, n_features).
    - sigma (float): Maximum noise level to be added to each feature.

    Returns:
    - np.ndarray: Simulated data matrix of the same shape as X with added noise, location shifts, and scaling.
    """
    
    # Generate noise for each feature
    noise = np.random.uniform(0.05, sigma, X.shape[1]).reshape(1, -1)
    
    # Generate location shifts (mean offsets) for each feature
    loc_ = np.random.uniform(-10, 10, X.shape[1]).reshape(1, -1)
    
    # Generate scaling factors for each feature
    scale_ = np.random.uniform(-5, 5, X.shape[1]).reshape(1, -1)
    
    # Apply noise, scaling, and location shifts to the input data
    X_simu = (X + noise) * scale_ + loc_
    
    return X_simu  


def simulate_normal_bimodal_matrix(n_samples, n_features):
    """
    Simulates a matrix with a bimodal distribution for each feature.

    Parameters:
    - n_samples (int): Number of samples in the matrix.
    - n_features (int): Number of features in the matrix.

    Returns:
    - np.ndarray: A numpy array of shape (n_samples, n_features) where each feature is bimodally distributed.
    """
    
    # Proportion of samples in the first mode
    proportion = np.random.uniform(0.2, 0.8)
    
    # Parameters for the second mode
    mu2 = np.random.uniform(-2, 2)     # Mean of the second mode
    std2 = np.random.uniform(1, 3)     # Standard deviation of the second mode

    # Calculate the number of samples in each mode
    n_samples_mode1 = int(n_samples * proportion)
    n_samples_mode2 = n_samples - n_samples_mode1

    # Initialize the matrix
    X = np.zeros((n_samples, n_features))

    for idx in range(n_features):
        # Generate data for the two modes
        x1 = np.random.normal(0, 1, n_samples_mode1)  # Data for the first mode
        x2 = np.random.normal(mu2, std2, n_samples_mode2)  # Data for the second mode
        
        # Combine the data from both modes
        x_combined = np.concatenate([x1, x2])

        # Standardize the combined data
        mu_combined = np.mean(x_combined)
        std_combined = np.std(x_combined)
        
        # Check for zero standard deviation to avoid division by zero
        x_scaled = (x_combined - mu_combined) / std_combined
        
        # Assign the standardized data to the feature column
        X[:, idx] = x_scaled
    
    return X


def simulate_binary_data(X, flip_rate):
    """
    Simulates binary data by generating a binary matrix from an input matrix and adding noise.

    Parameters:
    - X (np.ndarray): Input matrix of shape (n_samples, n_features) to generate binary data from.
    - flip_rate (float): Probability of flipping each binary value (0 to 1).

    Returns:
    - np.ndarray: Simulated binary data matrix of the same shape as X.
    """
    
    # Generate a threshold matrix with the same number of features as X
    threshold = np.random.uniform(-1.2, 1.2, X.shape[1]).reshape(1, -1)
    
    # Create binary data by comparing X to the threshold
    X_binary = (X > threshold).astype(int)
    
    # Generate a noise matrix with values between 0 and 1 of the same shape as X
    noise_matrix = np.random.uniform(0, 1, X.shape)
    
    # Flip binary data based on the noise matrix
    # If noise_matrix < flip_rate, flip the value; otherwise, keep the original value
    X_simu = np.where(noise_matrix < flip_rate, 1 - X_binary, X_binary)
    
    return X_simu


def simulate_count_data_poisson(X):
    """
    Simulates count data using a Poisson distribution based on the input matrix X.

    Parameters:
    - X (np.ndarray): Input matrix of shape (n_samples, n_features) used to compute Poisson rate parameters.

    Returns:
    - np.ndarray: Simulated count data matrix of the same shape as X, with Poisson-distributed values.
    """
    
    # Define a base value to be added to the input matrix
    base = 5
    
    # Calculate the mean (lambda) for the Poisson distribution
    # Use the exponential function to ensure positive mean values
    mu = np.exp(X + base)
    
    # Generate Poisson-distributed count data with the calculated mean
    X_simu = np.random.poisson(mu, X.shape)
    
    return X_simu


def simulate_count_data_negative_binomial(X, c):
    """
    Simulates count data using a Negative Binomial distribution based on the input matrix X.

    Parameters:
    - X (np.ndarray): Input matrix of shape (n_samples, n_features) used to compute the mean of the distribution.
    - c (float): Dispersion parameter that controls the variance of the distribution (should be > 1).

    Returns:
    - np.ndarray: Simulated count data matrix of the same shape as X, with Negative Binomial-distributed values.
    """
    
    # Ensure c is greater than 1 to avoid invalid calculations
    if c <= 1:
        raise ValueError("The parameter 'c' must be greater than 1.")
    
    # Define a base value to be added to the input matrix
    base = 5
    
    # Calculate the mean (lambda) for the Negative Binomial distribution
    mu = np.exp(X + base)
    
    # Calculate the probability of success (p) and number of successes (n)
    p = 1 / c
    n = mu / (c - 1)
    
    # Generate Negative Binomial-distributed count data
    X_simu = np.random.negative_binomial(n, p, size=X.shape)
    
    return X_simu


def simulate_uniform_data(X, epsilon):
    """
    Simulates data with a uniform noise component based on an input matrix X.

    Parameters:
    - X (np.ndarray): Input matrix of shape (n_samples, n_features) to be transformed.
    - epsilon (float): The range of uniform noise to be added.

    Returns:
    - np.ndarray: Simulated data matrix of the same shape as X, with added uniform noise and scaling.
    """
    
    # Generate random location parameters for scaling
    loc_ = np.random.uniform(-10, 10, X.shape[1]).reshape(1, -1)
    
    # Generate random scale parameters
    scale_ = np.random.uniform(-10, 10, X.shape[1]).reshape(1, -1)
    
    # Generate uniform noise
    uniform_noise = np.random.uniform(-epsilon, epsilon, X.shape)
    
    # Apply the cumulative distribution function transformation and add noise
    X_transformed = norm.cdf(X) + uniform_noise
    
    # Scale and shift the transformed data
    X_simu = X_transformed * scale_ + loc_
    
    return X_simu


def generate_labels(X_base, n_interact, label, noise):
    """
    Generates labels for regression or classification based on input features.

    Parameters:
    - X_base (np.ndarray): Feature matrix.
    - n_interact (int): Number of interaction terms
    - label (str): Type of label to generate ('regression' or 'classification').
    - noise (float): Noise level for regression or probability scaling for classification.

    Returns:
    - pd.Series: Generated labels.
    - pd.DataFrame: Coefficients used for generating labels.
    """
    # Generate n random interaction terms
    X_inter = pd.DataFrame(index=X_base.index)
    for _ in range(n_interact):
        # Randomly choose two distinct columns
        col1, col2 = np.random.choice(X_base.columns, size=2, replace=False)
        # Compute their product
        X_inter[f"{col1}*{col2}"] = X_base[col1] * X_base[col2]
    
    X = pd.concat([X_base, X_inter], axis=1)

    # Sample coefficients from the range (0.5, 1) and randomly multiply by -1 or 1
    coef = np.random.uniform(0.5, 1, X.shape[1])
    coef *= np.random.choice([-1, 1], size=X.shape[1])

    # Generate continuous labels using a combination of all feature types
    product = np.dot(X, coef) + np.random.normal(0, noise, X.shape[0])
    
    # Generate labels based on the type of problem
    if label == "regression":
        y = pd.Series(product, index=X.index, name="target")
    
    elif label == "classification":
        # Generate binary labels using a sigmoid function on the combined features
        y_prob = 1 / (1 + np.exp(-product))
        # Randomly sample a threshold for binary classification
        threshold = np.random.uniform(0.3, 0.7)
        # Apply threshold to generate binary labels
        y = pd.Series((y_prob > threshold).astype(int), index=X.index, name="target")
    
    else:
        raise ValueError("Invalid label type. Choose 'regression' or 'classification'.")
    
    # store coef
    coef = pd.DataFrame(coef.reshape(-1, 1), index=X.columns, columns=["coef"])
    
    return y, coef


if __name__ == "__main__":
    X, y, coef = simulate_data(
        n_samples=200, label="regression",
        n_feat_normal=10000, n_feat_normal_bimodal=0, 
        n_feat_count=0, n_feat_count_bimodal=0,
        n_feat_binary=0, n_feat_uniform=0, 
        n_base=6, n_interact=0, n_correlated=60,
        corr_level="high", noise_level_X="low", noise_level_y="low"
    )

    print(X)
    print(y)
    print(coef)