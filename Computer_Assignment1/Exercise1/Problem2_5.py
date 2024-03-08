from handout_Hws.src import *

# Set the random seed for reproducibility
np.random.seed(0)
# Create a one-hot vector with the random index
vector_size = 3
random_index = np.random.randint(0, vector_size - 1)
one_hot_vector = [0] * vector_size
one_hot_vector[random_index] = 1

# Generate data for the three classes
data_class1 = np.random.multivariate_normal(m1, cov_matrix, N // 3 + one_hot_vector[0])
data_class2 = np.random.multivariate_normal(m2, cov_matrix, N // 3 + one_hot_vector[1])
data_class3 = np.random.multivariate_normal(m3, cov_matrix, N // 3 + one_hot_vector[2])

# Seprate labels from data
labels_class1 = np.ones((N // 3 + one_hot_vector[0], 1))  # Label for class 1
labels_class2 = 2 * np.ones((N // 3 + one_hot_vector[1], 1))  # Label for class 2
labels_class3 = 3 * np.ones((N // 3 + one_hot_vector[2], 1))  # Label for class 3

# Combine data and labels for all classes
data = np.vstack((data_class1, data_class2, data_class3))
labels = np.vstack((labels_class1, labels_class2, labels_class3))

# Combine data and labels into a single array
combined_data = np.hstack((data, labels))

# Shuffle the dataset to mix the classes (use a fixed seed for reproducibility)
np.random.shuffle(combined_data)

# Separate the shuffled data and labels
shuffled_data = combined_data[:, :-1]  # Data without labels
shuffled_labels = combined_data[:, -1]  # Labels
X4=shuffled_data
true_labels=shuffled_labels
# Now, X4 contains 1,000 two-dimensional vectors with the specified properties

# Apply the classifiers to the dataset X4
bayesian_predictions = [Bayesian(x) for x in X4]
euclidean_predictions = [Euclidean(x) for x in X4]
mahalanobis_predictions = [Mahalanobis(x) for x in X4]

# Compute classification error for the Euclidean classifier
euclidean_error = sum(1 for pred, true in zip(euclidean_predictions, true_labels) if pred != true) / len(X4)

# Compute classification error for the Mahalanobis classifier
mahalanobis_error = sum(1 for pred, true in zip(mahalanobis_predictions, true_labels) if pred != true) / len(X4)

# Compute classification error for the Bayesian classifier
bayesian_error = sum(1 for pred, true in zip(bayesian_predictions, true_labels) if pred != true) / len(X4)

# Print the classification errors
print("Euclidean Classifier Error: {:.2f}%".format(euclidean_error * 100))
print("Mahalanobis Classifier Error: {:.2f}%".format(mahalanobis_error * 100))
print("Bayesian Classifier Error: {:.2f}%".format(bayesian_error * 100))

plot_data(X4,true_labels)




