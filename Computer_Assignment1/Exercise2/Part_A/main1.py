from src1 import *
# Define the column names for your dataset
column_names = ['Class', 'Feature1', 'Feature2', 'Feature3', 'Feature4']

# Load the dataset from the text file
data = pd.read_csv('Iris-Data_dat.txt', delim_whitespace=True, names=column_names)
# Separate the labels (target) and features
labels = data['Class']  # Extract the "Class" column as labels
features = data.drop(columns=['Class'])  # Remove the "Class" column to get the features
np_arr_features = features.values
N = len(np_arr_features)
NpC=50
print("Notice: value of g is Maximum Log Likelihood")
# Perform LOOCV for class separation
confusion_matrix = np.zeros((3, 3), dtype=int)
for i in range(N):
    X_train = np.delete(np_arr_features, i, axis=0)
    y_train = np.delete(labels.values, i)
    x_test = np_arr_features[i]
    y_test = labels.values[i]
    partition=np.zeros([3,1])
    if i<NpC-1:
        partition[0]=1
    elif i>=NpC-1 and i<2*NpC-1:
        partition[1]=1
    else:
        partition[2]=1
    ## Maximum Likelihood Estimation
    mu_1,cov_1=Max_likelihood_Estimation(X_train[0:NpC-1-int(partition[0][0])],NpC-int(partition[0][0]))
    mu_2,cov_2=Max_likelihood_Estimation(X_train[NpC-int(partition[0][0]):2*NpC-1-int(partition[1][0])],NpC-int(partition[1][0]))
    mu_3,cov_3=Max_likelihood_Estimation(X_train[2*NpC-int(partition[0][0])-int(partition[1][0]):N-int(partition[2][0])],NpC-int(partition[2][0]))
    ## Maximum log Likelihood Discriminant g
    pdf=np.zeros([1,3])
    pdf[0, 0] = multivariate_gaussian_likelihood(x_test, mu_1, cov_1)
    pdf[0, 1] = multivariate_gaussian_likelihood(x_test, mu_2, cov_2) 
    pdf[0, 2] = multivariate_gaussian_likelihood(x_test, mu_3, cov_3) 
    ## Prediction (Eval)
    predicted_class = np.argmax(pdf) + 1  # Class indices are 1, 2, 3
    # Calculate the confusion matrix
    confusion_matrix[y_test-1][predicted_class-1] += 1
    # Answer
    if y_test != predicted_class:
        print(f"Sample no.{i-(i//50 )*50} from class {y_test} incorrectly classified to class {predicted_class} with g{y_test}= {pdf[0,y_test-1]} and g{predicted_class}= {pdf[0,predicted_class-1]}")
        
plotconfusionmatrix(confusion_matrix)
