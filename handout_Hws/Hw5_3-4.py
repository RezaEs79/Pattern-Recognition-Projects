from src import *

# Input data
X_class1 = np.array([[0, 0], [0, 1]])
X_class2 = np.array([[1, 0], [1, 1]])
X = np.vstack([X_class1, X_class2])
# Labels for class 1 and class 2
class1_labels = np.ones(X_class1.shape[0])
class2_labels = -1 * np.ones(X_class2.shape[0])
y = np.hstack([class1_labels, class2_labels])
# y = np.array([1, 1, -1, -1])

# Run the perceptron algorithm
w, errors = perceptron_algorithm(X, y, pr='on')
if w[1] != 0 or w[0]!=0:
    print("\nNo samples changed the vector of weights => the algorithm was fixed and the linear classifier was found")
# Plot the data points and the decision boundary
show_result(X_class1,X_class2,w)


