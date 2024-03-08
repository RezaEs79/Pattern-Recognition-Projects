from src import *
## Extract Data
# Read Train Dataset
Training_data_dict = get_training()
# Read Test Dataset
Testing_data_dict = get_testing()
# Extract data for Initial Values of the Weights
w = get_weights()
matrix_input_to_hidden = w[0]
matrix_hidden_to_output = w[1]

## Train the MLP
X_train = np.column_stack((Training_data_dict['x1'],Training_data_dict['x2']))
y_train = np.column_stack((Training_data_dict['Target 1'],Training_data_dict['Target 2']))
trained_weights = train_mlp(X_train, y_train,  matrix_input_to_hidden, matrix_hidden_to_output)

## Test the MLP
X_test = np.column_stack((Testing_data_dict['x1'],Testing_data_dict['x2']))
predictions = test_mlp(X_test, trained_weights)

## Plot and Print Results
confusion_matrix = np.zeros((2, 2), dtype=int)
y_test=Testing_data_dict['Class']
# Calculate the confusion matrix
for i in range(len(y_test)):
    confusion_matrix[y_test[i]-1][predictions[i]-1] += 1
    if y_test[i] != predictions[i]:
        print(f"Sample no. {i} from class {y_test[i]}")
plotconfusionmatrix(confusion_matrix)