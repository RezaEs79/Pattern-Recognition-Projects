import numpy as np
import matplotlib.pyplot as plt
# Parameters for the three classes
N = 1000  # Number of samples
m1 = np.array([1, 1])  # Mean vector for class 1
m2 = np.array([10, 5])  # Mean vector for class 2
m3 = np.array([11, 1])  # Mean vector for class 3
cov_matrix = np.array([[7, 4], [4, 5]])  # Common covariance matrix for all classes
# Euclidean Classifier
def Euclidean(x):
    euclidean_dist1=((x[0] - m1[0]) ** 2 + (x[1] - m1[1]) ** 2)
    euclidean_dist2=((x[0] - m2[0]) ** 2 + (x[1] - m2[1]) ** 2)
    euclidean_dist3=((x[0] - m3[0]) ** 2 + (x[1] - m3[1]) ** 2)
    distances = [euclidean_dist1, euclidean_dist2, euclidean_dist3]
    predicted_class = np.argmin(distances) + 1  # Class indices are 1, 2, 3
    return predicted_class

# Mahalanobis Classifier
def Mahalanobis(x):
    mahalanobis_dist1 = np.dot(np.dot((x-m1), np.linalg.inv(cov_matrix)),(x-m1).T)
    mahalanobis_dist2 = np.dot(np.dot((x-m2), np.linalg.inv(cov_matrix)),(x-m2).T)
    mahalanobis_dist3 = np.dot(np.dot((x-m3), np.linalg.inv(cov_matrix)),(x-m3).T)
    distances = [mahalanobis_dist1, mahalanobis_dist2, mahalanobis_dist3]
    predicted_class = np.argmin(distances) + 1  # Class indices are 1, 2, 3
    return predicted_class

# Bayesian Classifier
def Bayesian(x):
    prior1,prior2,prior3=1/3,1/3,1/3
    discriminant1 = -0.5 * np.dot(np.dot((x-m1), np.linalg.inv(cov_matrix)),(x-m1).T) - 0.5 * np.log(np.linalg.det(cov_matrix)) + np.log(prior1)
    discriminant2 = -0.5 * np.dot(np.dot((x-m2), np.linalg.inv(cov_matrix)),(x-m2).T) - 0.5 * np.log(np.linalg.det(cov_matrix)) + np.log(prior2)
    discriminant3 = -0.5 * np.dot(np.dot((x-m3), np.linalg.inv(cov_matrix)),(x-m3).T) - 0.5 * np.log(np.linalg.det(cov_matrix)) + np.log(prior3)
    discriminants = [discriminant1, discriminant2, discriminant3]
    predicted_class = np.argmax(discriminants) + 1  # Class indices are 1, 2, 3
    return predicted_class

def plot_data(data,true_labels):
    # Define colors for each class
    colors = {1: 'red', 2: 'green', 3: 'blue'}

    # Create a 2D scatter plot with different colors for each class
    plt.figure(figsize=(8, 6))

    for class_label in np.unique(true_labels):
        class_indices = np.where(true_labels == class_label)
        plt.scatter(
            data[class_indices, 0],
            data[class_indices, 1],
            c=colors[class_label],
            marker='o',
            s=10,
            label=f'Class {int(class_label)}'
        )

    # Set labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('2D Scatter Plot of Dataset X4')

    # Show the plot
    plt.legend()
    plt.grid(True)
    plt.show()