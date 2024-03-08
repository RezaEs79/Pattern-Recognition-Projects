import pandas as pd
import numpy as np


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


############ Usage ##################



def K_NN_classification(X,y,option='a'):
    if option == 'MMD':
        # print("################################ Part B ###########################")
        correct_predictions = leave_one_out_cross_validation_MMD(X, y)
        accuracy = correct_predictions / len(X)
        return accuracy
        # print(f"MMD Classifier with Leave-One-Out for 'dataset' , Accuracy: % {accuracy*100:.2f}")
    else:
        # print("################################ Part A ###########################")
        k_values = [1, 3 ,5 ,7, 9]
        # print(f"The k-NN classifier with K= {k_values} and leave-one-out method ")
                
        # Create a mapping dictionary
        label_mapping = {'Kecimen': 1, 'Besni': 2}
        # Use the mapping to create a new vector
        new_labels = np.array([label_mapping[label] for label in y])
        accuracy_dict = {}
        for k in k_values:
            correct_predictions = leave_one_out_cross_validation(X, new_labels, [k])
            accuracy = correct_predictions[k] / len(X)
            # print(f"k={k}, Accuracy: % {accuracy *100:.2f}")
            accuracy_dict[k] = accuracy

    accuracy = sum(accuracy_dict.values()) / len(accuracy_dict)
    return accuracy