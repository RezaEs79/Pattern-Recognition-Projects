import pandas as pd
import numpy as np

def load_iris_dataset():
    # Define the column names for your dataset
    column_names = ['Class', 'Feature1', 'Feature2', 'Feature3', 'Feature4']
    # Load the dataset from the text file
    data = pd.read_csv('Data/Iris-Data_dat.txt', delim_whitespace=True, names=column_names)
    # Separate the labels (target) and features
    labels = data['Class']  # Extract the "Class" column as labels
    features = data.drop(columns=['Class'])  # Remove the "Class" column to get the features
    return features.values, labels.values

def load_liquid_dataset():
    # Define the column names for your dataset
    column_names = ['Class', 'Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6']

    # Load the dataset from the text file
    data = pd.read_csv('Data/Liquid-Data_dat.txt', delim_whitespace=True, names=column_names)
    # Separate the labels (target) and features
    labels = data['Class']  # Extract the "Class" column as labels
    features = data.drop(columns=['Class'])  # Remove the "Class" column to get the features
    return features.values, labels.values

def load_normal_dataset(inp):
    # Define the column names for your dataset
    column_names = ['Class', 'Feature1', 'Feature2']
    ###################################### Load the dataset from the train file
    if inp == 'train':
        data = pd.read_csv('Data/Normal-Data-Training_dat.txt', delim_whitespace=True, names=column_names)
    elif inp == 'test':
        data = pd.read_csv('Data/Normal-Data-Testing_dat.txt', delim_whitespace=True, names=column_names)
    else:
        raise ValueError("Invalid input")
    # Separate the labels (target) and features
    labels = data['Class']  # Extract the "Class" column as labels
    features = data.drop(columns=['Class'])  # Remove the "Class" column to get the features
    return features.values, labels.values

################################################# Part A ###########################################

# Calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

# Implement kNN classifier
def knn_classifier(X_train, y_train, x_test, k):
    distances = [euclidean_distance(x_test, x_train) for x_train in X_train]
    indices = np.argsort(distances)[:k]
    k_nearest_labels = y_train[indices]
    return np.bincount(k_nearest_labels).argmax()

# Leave-One-Out cross-validation
def leave_one_out_cross_validation(X, y, k_values):
    correct_predictions = {k: 0 for k in k_values}

    for i in range(len(X)):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)

        for k in k_values:
            prediction = knn_classifier(X_train, y_train, X[i], k)
            if prediction == y[i]:
                correct_predictions[k] += 1

    return correct_predictions


################################################# Part B ###########################################

# Calculate the mean of each class
def calculate_class_means(X, y):
    class_means = {}
    unique_classes = np.unique(y)

    for class_label in unique_classes:
        class_indices = np.where(y == class_label)
        class_means[class_label] = np.mean(X[class_indices], axis=0)

    return class_means

# Minimum Mean Distance (MMD) classifier
def mmd_classifier(class_means, x_test):
    distances = {class_label: np.linalg.norm(x_test - class_means[class_label]) for class_label in class_means}
    return min(distances, key=distances.get)

# Leave-One-Out cross-validation
def leave_one_out_cross_validation_MMD(X, y):
    correct_predictions = 0

    for i in range(len(X)):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)

        class_means = calculate_class_means(X_train, y_train)
        prediction = mmd_classifier(class_means, X[i])

        if prediction == y[i]:
            correct_predictions += 1

    return correct_predictions

# Test-dataset-validation
def Test_dataset_validation(X_train, y_train, X_test, y_test):
    correct_predictions = 0
    class_means = calculate_class_means(X_train, y_train)

    for i in range(len(X_test)):
        prediction = mmd_classifier(class_means, X_test[i])

        if prediction == y_test[i]:
            correct_predictions += 1

    return correct_predictions