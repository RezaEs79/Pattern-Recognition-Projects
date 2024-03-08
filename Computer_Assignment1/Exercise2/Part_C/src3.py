import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
## Maximum Likelihood Estimation
def Max_likelihood_Estimation(np_arr_features,Number):
    mu_hat = np_arr_features.sum(axis=0)/Number
    covariance_matrices = np.sum([np.outer((x - mu_hat).T, (x - mu_hat)) for x in np_arr_features], axis=0)/Number
    return mu_hat,covariance_matrices
# Compute the Mahalanobis distance for each data point
def mahalanobis_distance(diff,covariance_matrix):
    mahalanobis_distance = np.sqrt((diff).T @ np.linalg.inv(covariance_matrix) @ (diff))
    return mahalanobis_distance

def multivariate_gaussian_likelihood(x, mean, cov):
    p=1/3
    # Calculate the Mahalanobis distance
    diff = x - mean
    mahalanobis_distance = mahalanobis_distance(diff,cov)

    # Calculate the exponent term
    exponent_term = -0.5 * mahalanobis_distance

    # Constant term
    C = -0.5 * np.log(np.linalg.det(cov)) - (len(mean)/2)*np.log(2*np.pi)

    # Calculate the log likelihood
    log_likelihood = exponent_term + C + np.log(p)

    return log_likelihood

def find_misclassified(pred,y_test):
    misclassified_samples=[]
    for i in range(len(pred)):
        if y_test[i] != pred[i]:
            misclassified_samples.append(i + 1)
    return misclassified_samples

def build_confusion_matrix(pred, y_test):
    cmat=np.zeros((2, 2), dtype=int)
    # Num. misclassified sample for each class
    for i in range(len(pred)):
        cmat[y_test[i]-1][pred[i]-1]+=1
    return cmat


def err_each_class(pred, y_test):
    misclassified_samples = {} # Num. misclassified samples for each class ex. {1: 32, 2: 35}
    class_counts = {}  # Num. samples for each class ex. {1: 500, 2: 35}
    # Num. misclassified sample for each class
    for i in range(len(pred)):
        if y_test[i] != pred[i]:
            misclassified_samples.setdefault(y_test[i], 0)
            misclassified_samples[y_test[i]] += 1

        class_counts.setdefault(y_test[i], 0)
        class_counts[y_test[i]] += 1
    # Num. misclassified sample for each class ex. {1: 0.064, 2: 0.07}
    error_rate_per_class = {}
    for class_label, misclassified_count in misclassified_samples.items():
        class_count = class_counts[class_label]
        error_rate_per_class[class_label] = misclassified_count / class_count

    # return error_rate_per_class 
    # Change to str
    output_string = ""
    for label, error_rate in error_rate_per_class.items():
        output_string += f"error_ class{label}: {error_rate:.3f}, "
    output_string += f"and error_total is {sum(error_rate_per_class.values()):.5f}"
    output_string = output_string[:-2]
    return output_string

def plot_results(X_train, y_train, X_test, y_test, y_pred):
    # Create a 2D scatter plot with different colors for each class
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], marker='o', s=4, color='tab:blue', label='Class 1')
    plt.scatter(X_train[y_train == 2, 0], X_train[y_train == 2, 1], marker='s', s=4, color='skyblue', label='Class 2')
    plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], marker='o', s=4, color='tab:blue')
    plt.scatter(X_test[y_test == 2, 0], X_test[y_test == 2, 1], marker='s', s=4, color='skyblue')

    label_added_class_1 = False
    label_added_class_2 = False

    for i in range(len(X_test)):
        if y_test[i] != y_pred[i]:
            color = 'magenta' if (y_test[i] == 1) else 'red'
            marker = 'x' if (y_test[i] == 1) else '+'
            label = 'Misclassified (Class 1)' if (y_test[i] == 1) and not label_added_class_1 else 'Misclassified (Class 2)' if (y_test[i] == 2) and not label_added_class_2 else None
            plt.scatter(X_test[i, 0], X_test[i, 1], s=9, c=color, marker=marker, label=label)
            if label == 'Misclassified (Class 1)':
                label_added_class_1 = True
            elif label == 'Misclassified (Class 2)':
                label_added_class_2 = True

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Training and Testing Samples with Misclassifications')
    plt.legend()
    plt.show()
