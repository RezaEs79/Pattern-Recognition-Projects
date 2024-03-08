import numpy as np
import pandas as pd
from tree import Tree

NUMBER_OF_ATTEMPTS = 3

# all important things happen here
def print_accuracy_of_tree(train_dataframe, test_dataframe, ratio_of_train, number_of_attempts=3):
    print("----------------")
    print("using ", np.round(ratio_of_train*100), "% of train data:")
    # convert dataframes to numpy array
    test_array = test_dataframe.to_numpy(dtype=np.string_)
    # compute sum of all results to compute average
    sum_of_results_test = 0
    sum_of_results_train = 0
    for i in range(number_of_attempts):
        # select entered ratio of train datasets randomly
        train_dataframe_ratio = train_dataframe.sample(frac=ratio_of_train)

        # convert train_dataframe to numpy array
        train_array = train_dataframe_ratio.to_numpy(dtype=np.string_)

        # train tree
        my_tree = Tree(train_array[:, 1:], train_array[:, 0])
        my_tree.fit()

        # compute results of tree for test_array
        predicted_y_test = my_tree.predict(np.copy(test_array[:,1:]))
        predicted_y_train = my_tree.predict(np.copy(train_array[:,1:]))

        # compute accuracy of test data
        accuracy = Tree.compute_accuracy(predicted_y_test, test_array[:, 0])
        sum_of_results_test += accuracy

        # compute accuracy of train data
        accuracy2 = Tree.compute_accuracy(predicted_y_train, train_array[:, 0])
        sum_of_results_train += accuracy2


        # print accuracy
        print("attempt number ", i+1,": test accuracy = ", accuracy, ", number of nodes: ", my_tree.number_of_nodes(my_tree.root))
        print("train accuracy = ", accuracy2)
    # print average
    print("average accuracy train: ", sum_of_results_train / number_of_attempts)
    print("average accuracy: ", sum_of_results_test / number_of_attempts)
# read data
df_train = pd.read_csv("adult.train.10k.discrete", header=None)
df_test = pd.read_csv("adult.test.10k.discrete", header=None)

# compute accuracy of tree for different portions of train_set
print_accuracy_of_tree(df_train, df_test, 0.45, number_of_attempts=NUMBER_OF_ATTEMPTS)
print_accuracy_of_tree(df_train, df_test, 0.55, number_of_attempts=NUMBER_OF_ATTEMPTS)
print_accuracy_of_tree(df_train, df_test, 0.65, number_of_attempts=NUMBER_OF_ATTEMPTS)
print_accuracy_of_tree(df_train, df_test, 0.75, number_of_attempts=NUMBER_OF_ATTEMPTS)
print_accuracy_of_tree(df_train, df_test, 1, number_of_attempts=NUMBER_OF_ATTEMPTS)