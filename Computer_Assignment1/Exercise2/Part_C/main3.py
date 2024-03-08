from src3 import *
# Define the column names for your dataset
column_names = ['Class', 'Feature1', 'Feature2']
###################################### Load the dataset from the train file
data_train = pd.read_csv('Normal-Data-Training_dat.txt', delim_whitespace=True, names=column_names)
# Separate the labels (target) and features
labels_train = data_train['Class']  # Extract the "Class" column as labels
features_train = data_train.drop(columns=['Class'])  # Remove the "Class" column to get the features
np_arr_features_train = features_train.values
###################################### Load the dataset from the test file
data_test = pd.read_csv('Normal-Data-Testing_dat.txt', delim_whitespace=True, names=column_names)
# Separate the labels (target) and features
labels_test = data_test['Class']  # Extract the "Class" column as labels
features_test = data_test.drop(columns=['Class'])  # Remove the "Class" column to get the features
np_arr_features_test = features_test.values
#
N = len(np_arr_features_train)
NpC = 500
first_500_rows = np_arr_features_train[:500, :]  # Rows 0 to 499, all columns
last_500_rows = np_arr_features_train[500:, :]   # Rows 500 to 999, all columns
# Maximum Likelihood Estimation
mu_1,cov_1=Max_likelihood_Estimation(first_500_rows,NpC)
mu_2,cov_2=Max_likelihood_Estimation(last_500_rows,NpC)
confusion_matrix = np.zeros((2, 2), dtype=int)
dist1 = [multivariate_gaussian_likelihood(x,mu_1,cov_1) for x in np_arr_features_test]
dist2 = [multivariate_gaussian_likelihood(x,mu_2,cov_2) for x in np_arr_features_test]
comparison = np.where(np.array(dist1) > np.array(dist2), 1, 2)
# Results
plot_results(np_arr_features_train,labels_train, np_arr_features_test,labels_test, comparison)
print(f"\n-List all the misclassified samples:  {find_misclassified(comparison,labels_test)}")
print("\n-*** "+err_each_class(comparison,labels_test)+" ***\n")
build_confusion_matrix(comparison, labels_test)


