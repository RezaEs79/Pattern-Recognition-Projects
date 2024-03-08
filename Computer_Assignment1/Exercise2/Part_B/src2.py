import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
## Maximum Likelihood Estimation
def Max_likelihood_Estimation(np_arr_features,Number):
    mu_hat = np_arr_features.sum(axis=0)/Number
    covariance_matrices = np.sum([np.outer((x - mu_hat).T, (x - mu_hat)) for x in np_arr_features], axis=0)/Number
    return mu_hat,covariance_matrices

def multivariate_gaussian_likelihood(x, mean, cov):
    p=1/3
    # Calculate the Mahalanobis distance
    diff = x - mean
    mahalanobis_distance = diff @ np.linalg.inv(cov) @ diff

    # Calculate the exponent term
    exponent_term = -0.5 * mahalanobis_distance

    # Constant term
    C = -0.5 * np.log(np.linalg.det(cov)) - (len(mean)/2)*np.log(2*np.pi)

    # Calculate the log likelihood
    log_likelihood = exponent_term + C + np.log(p)

    return log_likelihood

def plotconfusionmatrix(confusion_matrix):
    # Create a custom confusion matrix plot
    plt.figure(figsize=(8, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    # Label the axes
    tick_marks = np.arange(3)
    plt.xticks(tick_marks, [f"Predicted {i+1}" for i in range(3)])
    plt.yticks(tick_marks, [f"True {i+1}" for i in range(3)])

    # Display the values in the matrix
    # Display the values in the matrix with custom text color
    for i in range(3):
        for j in range(3):
            color = 'black' if confusion_matrix[i][j] < np.max(confusion_matrix) / 2 else 'white'
            plt.text(j, i, str(confusion_matrix[i][j]), ha='center', va='center', color=color)

    plt.xlabel('Predicted', fontsize=16, fontweight='bold')
    plt.ylabel('True', fontsize=16, fontweight='bold')

    plt.show()