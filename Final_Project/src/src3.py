import numpy as np
import copy
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def train_mlp(X_train, y_train, W1, W2, epochs=1000, learning_rate=0.01):
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
            hidden_output_bias = np.column_stack((hidden_output, np.ones((hidden_output.shape[0], 1))))
            input_of_output = np.dot(hidden_output_bias, weights_hidden_output)
            predicted_output = sigmoid(input_of_output)

            # Backward pass
            output_error = target_data - predicted_output
            output_delta = output_error * sigmoid_derivative(predicted_output)

            # Perform matrix multiplication
            hidden_layer_error = np.dot(output_delta, weights_hidden_output[:-1, :].T)
            hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_output)

            # Update
            weights_hidden_output += learning_rate * np.dot(hidden_output_bias.T, output_delta)
            weights_input_hidden += learning_rate * np.dot(input_data.T, hidden_layer_delta)

    return weights_input_hidden, weights_hidden_output

def test_mlp(X_test, trained_weights):
    # Add bias to input layer
    X_test = np.column_stack((X_test, np.ones((X_test.shape[0], 1))))
    
    # Forward pass
    hidden_input = np.dot(X_test, trained_weights[0])
    hidden_output = sigmoid(hidden_input)
    
    # Add bias to hidden layer
    hidden_output_bias = np.column_stack((hidden_output, np.ones((hidden_output.shape[0], 1))))
    
    output_layer_input = np.dot(hidden_output_bias, trained_weights[1])
    predicted_output = sigmoid(output_layer_input)
    
    # Classify based on higher output value
    predictions = np.argmax(predicted_output, axis=1)
    return predictions + 1


def MLP_classification(np_arr_features,np_arr_labels):
    # Set a random seed for reproducibility
    np.random.seed(42)

    # Create a mapping dictionary
    class_mapping = {'Kecimen': [0.95, 0.05], 'Besni': [0.05, 0.95]}

    # Use the mapping to create a new vector
    y_map = np.array([class_mapping[label] for label in np_arr_labels])

    # Create random weight matrices with appropriate shapes
    matrix_input_to_hidden = np.random.randn(np_arr_features.shape[1]+1, 14) * 0.1
    matrix_hidden_to_output = np.random.randn(15, 2) * 0.1 # Corrected dimension

    # Min-Max normalization for each column separately
    new_min = 0
    new_max = 10

    min_values = np.min(np_arr_features, axis=0)
    max_values = np.max(np_arr_features, axis=0)

    X_norm = (np_arr_features - min_values) / (max_values - min_values) * (new_max - new_min) + new_min

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y_map, test_size=0.2, random_state=42)

    # Train the MLP
    trained_weights = train_mlp(X_train, y_train, matrix_input_to_hidden, matrix_hidden_to_output)

    # Test the MLP
    predictions = test_mlp(X_test, trained_weights)

    # Create a new array with the desired mapping
    y_test_binary = np.where(np.all(y_test == [0.95, 0.05], axis=1), 1, 2)

    # Calculate the accuracy
    accuracy = np.mean(predictions == y_test_binary)

    return accuracy
    # Print the result
    # print(f"Accuracy: {accuracy * 100:.2f}%")