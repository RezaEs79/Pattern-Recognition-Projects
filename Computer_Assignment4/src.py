import numpy as np
import copy
import matplotlib.pyplot as plt
import pandas as pd

# Replace 'path/to/Computer_Assignment_4_Data.xls' with the actual path to your file
file_path = 'Computer_Assignment_4_Data.xls'

# Read Excel file into a pandas DataFrame
def get_training():
    df = pd.read_excel(file_path, sheet_name = 'Training Set',skiprows=1)
    # Create a dictionary to store columns
    Training_data_dict = {
        'Class': df['Class #'],
        'x1': df['x1'],
        'x2': df['x2'],
        'Target 1': df['Target 1'],
        'Target 2': df['Target 2']
    }
    return Training_data_dict 

def get_testing():
    df = pd.read_excel(file_path, sheet_name = 'Testing Set',skiprows=1)

    # Create a dictionary to store columns
    Testing_data_dict = {
        'Class': df['Class #'],
        'x1': df['x1'],
        'x2': df['x2']
    }
    return Testing_data_dict
def get_weights():
    df = pd.read_excel(file_path, header=None, sheet_name="Initial Weights",skiprows=4)
    input_to_hidden_weights = df.iloc[:12, 2].values
    matrix_weight_1 = input_to_hidden_weights.reshape(3, 4)
    hidden_to_output_weights = df.iloc[:10, 6].values
    matrix_weight_2 = hidden_to_output_weights.reshape(5, 2)
    return matrix_weight_1, matrix_weight_2

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

def test_mlp(X_test, trained_weights):
    # Add bias to input layer
    X_test = np.column_stack((X_test, np.ones((X_test.shape[0], 1))))
    
    # Forward pass
    hidden_input = np.dot(X_test, trained_weights[0])
    hidden_output = sigmoid(hidden_input)
    
    # Add bias to hidden layer
    hidden_output = np.column_stack((hidden_output, np.ones((hidden_output.shape[0], 1))))
    
    output_layer_input = np.dot(hidden_output, trained_weights[1])
    predicted_output = sigmoid(output_layer_input)
    
    # Classify based on higher output value
    predictions = np.argmax(predicted_output, axis=1) + 1
    return predictions

def train_mlp(X_train, y_train, W1,W2, epochs=500, learning_rate=0.2):
    # Add bias to input and hidden layer
    X_train_bias = np.column_stack((X_train, np.ones((X_train.shape[0], 1))))
    weights_input_hidden = copy.deepcopy(W1)
    weights_hidden_output = copy.deepcopy(W2)

    for epoch in range(epochs):
        for i in range(X_train_bias.shape[0]):
            input_data = X_train_bias[i:i+1, :]
            target_data = y_train[i:i+1, :]
            # Forward pass
            hidden_input = np.dot(input_data, weights_input_hidden)
            hidden_output = sigmoid(hidden_input)
            
            # Add bias to hidden layer
            hidden_output_bias = np.column_stack((hidden_output, np.ones(hidden_output.shape[0])))
            input_of_output = np.dot(hidden_output_bias, weights_hidden_output)
            predicted_output = sigmoid(input_of_output)

            # Backward pass
            output_error = target_data - predicted_output
            output_delta = output_error * sigmoid_derivative(predicted_output)

            # Perform matrix multiplication
            hidden_layer_error = np.dot(output_delta, weights_hidden_output[:-1,:].T)
            hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_output)
            # Update
            weights_hidden_output +=learning_rate * np.dot(hidden_output_bias.T, output_delta)
            weights_input_hidden += learning_rate * np.dot(input_data.T, hidden_layer_delta)

    return weights_input_hidden, weights_hidden_output
                
def test_mlp(X_test, trained_weights):
    # Add bias to input layer
    X_test = np.column_stack((X_test, np.ones((X_test.shape[0], 1))))
    
    # Forward pass
    hidden_input = np.dot(X_test, trained_weights[0])
    hidden_output = sigmoid(hidden_input)
    
    # Add bias to hidden layer
    hidden_output = np.column_stack((hidden_output, np.ones((hidden_output.shape[0], 1))))
    
    output_layer_input = np.dot(hidden_output, trained_weights[1])
    predicted_output = sigmoid(output_layer_input)
    
    # Classify based on higher output value
    predictions = np.argmax(predicted_output, axis=1) + 1
    return predictions

def plotconfusionmatrix(confusion_matrix):
    # Create a custom confusion matrix plot
    plt.figure(figsize=(8, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    # Label the axes
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, [f"Predicted {i+1}" for i in range(2)])
    plt.yticks(tick_marks, [f"True {i+1}" for i in range(2)])

    # Display the values in the matrix
    # Display the values in the matrix with custom text color
    for i in range(2):
        for j in range(2):
            color = 'black' if confusion_matrix[i][j] < np.max(confusion_matrix) / 2 else 'white'
            plt.text(j, i, str(confusion_matrix[i][j]), ha='center', va='center', color=color)

    plt.xlabel('Predicted', fontsize=16, fontweight='bold')
    plt.ylabel('True', fontsize=16, fontweight='bold')

    plt.show()



