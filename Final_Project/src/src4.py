import numpy as np

def standardize_data(X):
    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)
    standardized_data = (X - mean) / std_dev
    return standardized_data, mean, std_dev

def calculate_covariance_matrix(X):
    covariance_matrix = np.cov(X, rowvar=False)
    return covariance_matrix

def calculate_eigenvalues_and_eigenvectors(covariance_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    # Sort eigenvalues and corresponding eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    return eigenvalues, eigenvectors

def explained_variance_ratio(eigenvalues):
    total_variance = np.sum(eigenvalues)
    explained_variance = eigenvalues / total_variance
    return explained_variance

def project_data(X, eigenvectors, num_components):
    projection_matrix = eigenvectors[:, :num_components]
    projected_data = np.dot(X, projection_matrix)
    return projected_data

def feature_selection(X_train, num_components):
    # Assuming X_train is your training data (features)
    # Make sure X_train has shape (number of samples, number of features)

    # Step 1: Standardize the data
    X_train_std, mean, std_dev = standardize_data(X_train)

    # Step 2: Calculate the covariance matrix
    cov_matrix = calculate_covariance_matrix(X_train_std)

    # Step 3: Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = calculate_eigenvalues_and_eigenvectors(cov_matrix)

    ## Step 4: Calculate explained variance ratio
    # explained_var = explained_variance_ratio(eigenvalues)
    # print("Explained Variance Ratio:", explained_var)

    # Step 5: Project the data onto principal components
    # num_components = 5  # You can adjust this based on your requirement
    X_train_pca = project_data(X_train_std, eigenvectors, num_components)
    return X_train_pca

# Now X_train_pca contains the transformed features after PCA
# You can use this in your machine learning model