import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
## Maximum Likelihood Estimation
def Max_likelihood_Estimation(np_arr_features,Number):
    mu_hat = np_arr_features.sum(axis=0)/Number
    covariance_matrices = np.sum([np.outer((x - mu_hat).T, (x - mu_hat)) for x in np_arr_features], axis=0)/Number
    return mu_hat,covariance_matrices

def multivariate_gaussian_likelihood(x, mean, cov):

    # Calculate the Mahalanobis distance
    diff = x - mean
    mahalanobis_distance = diff @ np.linalg.inv(cov) @ diff

    # Calculate the exponent term
    exponent_term = -0.5 * mahalanobis_distance

    # Constant term
    C = -0.5 * np.log(np.linalg.det(cov)) - (len(mean)/2)*np.log(2*np.pi)

    # Calculate the log likelihood
    log_likelihood = exponent_term + C + np.log(1/3)

    return log_likelihood

def plotconfusionmatrix(confusion_matrix):
    Numclass = 2
    # Create a custom confusion matrix plot
    plt.figure(figsize=(8, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    # Label the axes
    tick_marks = np.arange(Numclass)
    plt.xticks(tick_marks, [f"Predicted {i+1}" for i in range(Numclass)])
    plt.yticks(tick_marks, [f"True {i+1}" for i in range(Numclass)])

    # Display the values in the matrix with custom text color
    for i in range(Numclass):
        for j in range(Numclass):
            color = 'black' if confusion_matrix[i][j] < np.max(confusion_matrix) / 2 else 'white'
            plt.text(j, i, str(confusion_matrix[i][j]), ha='center', va='center', color=color)

    plt.xlabel('Predicted', fontsize=16, fontweight='bold')
    plt.ylabel('True', fontsize=16, fontweight='bold')

    plt.show()


def Bays_Classification(np_arr_features,labels,label_counts):
    # Perform LOOCV for class separation
    confusion_matrix = np.zeros((2, 2), dtype=int)
    N = len(np_arr_features)
    for i in range(N):
        X_train = np.delete(np_arr_features, i, axis=0)
        y_train = np.delete(labels.values, i)
        x_test = np_arr_features[i]
        y_test = labels.values[i]
        partition=np.zeros([2,1])
        if i<label_counts['Kecimen']-1:
            partition[0]=1
        else:
            partition[1]=1
        ## Maximum Likelihood Estimation
        mu_1,cov_1=Max_likelihood_Estimation(X_train[0:label_counts['Kecimen']-1-int(partition[0][0])],label_counts['Kecimen']-int(partition[0][0]))
        mu_2,cov_2=Max_likelihood_Estimation(X_train[label_counts['Kecimen']-int(partition[0][0]):label_counts['Kecimen']+label_counts['Besni']-1-int(partition[1][0])],label_counts['Besni']-int(partition[1][0]))
        ## Maximum log Likelihood Discriminant g
        pdf=np.zeros([1,2])
        pdf[0, 0] = multivariate_gaussian_likelihood(x_test, mu_1, cov_1)
        pdf[0, 1] = multivariate_gaussian_likelihood(x_test, mu_2, cov_2)  
        ## Prediction (Eval)
        predicted_class = np.argmax(pdf) + 1  # Class indices are 1, 2
        # Calculate the confusion Matrix
        if y_test == 'Kecimen':
            index_label = 1
        elif y_test == 'Besni':
            index_label = 2
        confusion_matrix[index_label-1][predicted_class-1] += 1

    
    main_diagonal_sum = np.trace(confusion_matrix)
    total_sum = np.sum(confusion_matrix)

    accuracy = main_diagonal_sum / total_sum
    return accuracy