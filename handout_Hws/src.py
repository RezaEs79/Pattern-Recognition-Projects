import numpy as np
import matplotlib.pyplot as plt
# Parameters for the three classes
N = 100  # Number of samples
m1 = np.array([1, 1])  # Mean vector for class 1
m2 = np.array([0, 0])  # Mean vector for class 2
cov_matrix = 0.2 * np.eye(2)  # Common covariance matrix for all classes

# Function to generate data for each class with conditions
def generate_data(m, cov, num_samples, condition):
    data = []
    while len(data) < num_samples:
        sample = np.random.multivariate_normal(m, cov, 1)[0]
        if condition(sample):
            data.append(np.concatenate([sample]))
    return np.array(data)




def generate_preprocess_data(m1, m2, cov_matrix, N, condition_class1,condition_class2):
    # Generate data for the two classes with conditions
    data_class1 = generate_data(m1, cov_matrix, N // 2, condition_class1)
    data_class2 = generate_data(m2, cov_matrix, N // 2, condition_class2)

    # Seprate labels from data
    labels_class1 = np.ones((N // 2, 1))  # Label for class 1
    labels_class2 = -1 * np.ones((N // 2, 1))  # Label for class 2

    # Combine data and labels for all classes
    data = np.vstack((data_class1, data_class2))
    labels = np.vstack((labels_class1, labels_class2))

    # Combine data and labels into a single array
    combined_data = np.hstack((data, labels))

    # Shuffle the dataset to mix the classes (use a fixed seed for reproducibility)
    np.random.shuffle(combined_data)
    
    # Separate the shuffled data and labels
    shuffled_data = combined_data[:, :-1]  # Data without labels
    shuffled_labels = combined_data[:, -1]  # Labels
    return shuffled_data,shuffled_labels,data_class1,data_class2

def perceptron_algorithm(X, y, rho=1, max_epochs=1000,pr='off'):
    """
    Perceptron algorithm in its reward and punishment form.

    Parameters:
    - X: Input features (numpy array with shape [num_samples, num_features])
    - y: Class labels (numpy array with shape [num_samples])
    - rho: Learning rate (default is 1)
    - max_epochs: Maximum number of epochs (default is 1000)
    - pr: print solution procedure or not

    Returns:
    - w: Learned weight vector
    - errors: List of errors at each epoch
    """
    num_samples, num_features = X.shape
    w = np.zeros(num_features + 1)  # Add bias term
    bias = 1
    X_bias = np.c_[X, bias*np.ones(num_samples)]  # Add bias term to features
    errors = []
    step =0
    con_final=1
    for epoch in range(max_epochs):
        error_count = 0
        for i in range(num_samples):
            x_i = X_bias[i]
            y_i = y[i]
            prediction = np.dot(w, x_i)
            if y_i * prediction <= 0:
                if pr=='on':
                    print(f"\nstep {step+1} \n\t w^T({step}) * x({step}) = {prediction} {' ':<15} w({step+1})=w({step}) {'+' if y_i >0 else '-'} \u03C1 x({step}) {' ':<15}",end="")
                w = w + rho * y_i * x_i
                if pr=='on':
                    print(f"w({step+1})=w({step}) {'+' if y_i >0 else '-'} \u03C1 {x_i.astype(int)} = {w.astype(int)}")
                error_count +=1
                con_final=1
            else:
                if pr=='on':
                    print(f"\nstep {step+1} : \n\t w^T({step}) * x({step}) = {prediction:<18}  w({step+1}) = w({step}) = {w.astype(int)} ")
                con_final=con_final+1
            step=step+1
        errors.append(error_count)
        if con_final>=num_samples-1:
            break
        # if error_count == 0:
        #     break

    return w, errors

def show_result(data_class1,data_class2,w):
    plt.scatter(data_class1[:, 0], data_class1[:, 1], label='Class 1', marker='o')
    plt.scatter(data_class2[:, 0], data_class2[:, 1], label='Class 2', marker='x')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Perceptron Algorithm - Decision Boundary')
    if w[1] == 0 and w[0]!=0:
        # Plot the vertical line
        plt.axvline(x=-(w[2]/w[0]), color='r', linestyle='--', label='Decision Boundary')
    elif w[1] != 0 and w[0]==0:
        # Plot the horizontal line
        plt.axhline(y=-(w[2]/w[1]), color='r', linestyle='--', label='Decision Boundary')
    elif w[1] != 0 and w[0]!=0:
        x_decision_boundary = np.linspace(-1, 2, 100)
        y_decision_boundary = -(w[0] * x_decision_boundary + w[2]) / w[1]
        plt.plot(x_decision_boundary, y_decision_boundary, color='r', linestyle='--', label='Decision Boundary')
    else:
        print('No Decision Boundary')
    plt.legend()
    plt.show()