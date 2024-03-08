import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# functions below to part 1
        
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
    plt.title('Decision Boundary')
    if w[1] == 0 and w[0]!=0:
        # Plot the vertical line
        plt.axvline(x=-(w[2]/w[0]), color='r', linestyle='--', label='Decision Boundary')
    elif w[1] != 0 and w[0]==0:
        # Plot the horizontal line
        plt.axhline(y=-(w[2]/w[1]), color='r', linestyle='--', label='Decision Boundary')
    elif w[1] != 0 and w[0]!=0:
        x_decision_boundary = np.linspace(1.2*min(min(data_class1[:, 0]),min(data_class1[:, 1])), 1.2*max(max(data_class1[:, 0]),max(data_class1[:, 1])), 100)
        y_decision_boundary = -(w[0] * x_decision_boundary + w[2]) / w[1]
        plt.plot(x_decision_boundary, y_decision_boundary, color='r', linestyle='--', label='Decision Boundary')
    else:
        print('No Decision Boundary')
    plt.legend()
    plt.show()
# functions below to part 2
def least_squares(X, y):
    # Add a column of ones to the feature matrix for the bias
    X = np.c_[X, np.ones(X.shape[0])]

    # Calculate the coefficients using the pseudoinverse
    coefficients = np.linalg.pinv(X.T @ X) @ X.T @ y

    return coefficients
# functions below to part 3
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, learning_rate=0.01, epochs=1000):
    # Add bias term to the features
    X = np.c_[X, np.ones((X.shape[0], 1))]
    w = np.zeros(X.shape[1])

    for epoch in range(epochs):
        z = np.dot(X, w)
        h = sigmoid(z)
        # gradient = np.dot(X.T, (h - y)) / len(y)
        gradient = np.dot(X.T, (h - y)) 
        w -= learning_rate * gradient

    return w

def one_vs_all(X, y, num_classes):
    all_w = []

    for i in range(num_classes):
        binary_labels = (y == (i+1)).astype(int) # vector of 1 and 0 where 1 is for data class i+1 and 0 for others
        w = train_logistic_regression(X, binary_labels)
        all_w.append(w)

    return np.array(all_w)

def one_vs_one(X, y, num_classes):
    all_Ws = []

    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            class_indices = np.logical_or(y == (i+1), y == (j+1)) # vector that is True where y=i+1 or y=j+1 and False for other places
            binary_labels = (y[class_indices] == (i+1)).astype(int) # vector of 1 and 0 where 1 is for data class i+1 and 0 for data class j+1
            binary_X = X[class_indices] # seprate data just for class i+1 and j+1
            W = train_logistic_regression(binary_X, binary_labels)
            all_Ws.append(((i+1), (j+1), W))

    return all_Ws

def plot_logsim_one_vs_all(concatenated_data,lines):
    # Separate the data into classes
    classes = np.unique(concatenated_data[:, 0])

    # Create a scatter plot for each class with different colors
    for cls in classes:
        class_data = concatenated_data[concatenated_data[:, 0] == cls]
        plt.scatter(class_data[:, 1], class_data[:, 2], label=f'Class {int(cls)}')

    # Add legend
    plt.legend()
    # Extracting coefficients from the matrix
    w1_values = lines[:, 0]
    w2_values = lines[:, 1]
    w0_values = lines[:, 2]

    # Generate x values for plotting
    x_values = np.linspace(1.1*min(min(concatenated_data[:, 1]),min(concatenated_data[:, 2])), 1.1*max(max(concatenated_data[:, 1]),max(concatenated_data[:, 2])), 100)

    # Plotting the lines
    for i in range(len(lines)):
        y_values = (-w0_values[i] - w1_values[i] * x_values) / w2_values[i]
        plt.plot(x_values, y_values, label=f'Line {i + 1}')

    # Set labels and title
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Plot Logistic Discrimination one_vs_all')
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)

    # Add legend
    plt.legend()
    # Set xlim and ylim
    plt.xlim(1.1*min(min(concatenated_data[:, 1]),min(concatenated_data[:, 2])), 1.1*max(max(concatenated_data[:, 1]),max(concatenated_data[:, 2])))
    plt.ylim(1.1*min(min(concatenated_data[:, 1]),min(concatenated_data[:, 2])), 1.1*max(max(concatenated_data[:, 1]),max(concatenated_data[:, 2])))
    # Show the plot
    plt.show()

def plot_logsim_one_vs_one(concatenated_data,lines_data):
    # Separate the data into classes
    classes = np.unique(concatenated_data[:, 0])

    # Create a scatter plot for each class with different colors
    for cls in classes:
        class_data = concatenated_data[concatenated_data[:, 0] == cls]
        plt.scatter(class_data[:, 1], class_data[:, 2], label=f'Class {int(cls)}')

    # Add legend
    plt.legend()
    # Generate x values for plotting
    x_values = np.linspace(1.1*min(min(concatenated_data[:, 1]),min(concatenated_data[:, 2])), 1.1*max(max(concatenated_data[:, 1]),max(concatenated_data[:, 2])), 100)

    # Plotting the lines
    for line_data in lines_data:
        class_label1, class_label2, coefficients = line_data
        w1, w2, w0 = coefficients
        y_values = (-w0 - w1 * x_values) / w2
        plt.plot(x_values, y_values, label=f'Class {class_label1} vs Class {class_label2}')

    # Set labels and title
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Plot Logistic Discrimination one_vs_one')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)

    # Add legend
    plt.legend()
    # Set xlim and ylim
    plt.xlim(1.1*min(min(concatenated_data[:, 1]),min(concatenated_data[:, 2])), 1.1*max(max(concatenated_data[:, 1]),max(concatenated_data[:, 2])))
    plt.ylim(1.1*min(min(concatenated_data[:, 1]),min(concatenated_data[:, 2])), 1.1*max(max(concatenated_data[:, 1]),max(concatenated_data[:, 2])))
    # Show the plot
    plt.show()