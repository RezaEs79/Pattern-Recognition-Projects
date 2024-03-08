import numpy as np
from numpy.lib.shape_base import array_split
import pandas as pd
from tree import Tree
import matplotlib.pyplot as plt

# read data
train_df = pd.read_csv("adult.train.10k.discrete")
test_df = pd.read_csv("adult.test.10k.discrete")

# convert data to numpy array
train_array = train_df.to_numpy(dtype=np.string_)
np.random.shuffle(train_array)
test_array = test_df.to_numpy(dtype=np.string_)

# start making trees and train them using k-folt
K = 4
trees = []
split_train_array_list = []
validation_array_list = []
for i in range(K):
    steps_size = int(train_array.shape[0] / K)
    split_train_array_list.append(np.delete(np.copy(train_array), range(steps_size*i, steps_size*(i+1)), axis=0))
    validation_array_list.append(train_array[steps_size*i:steps_size*(i+1)+1,:])
    trees.append(Tree(split_train_array_list[-1][:,1:], split_train_array_list[-1][:,0]))
    trees[-1].fit()

# compute accuracy on test and number of nodes before pruning
y_test_list = []
for tree in trees:
    y_test_list.append(tree.predict(test_array[:,1:]))
y_test_array = np.sum(np.concatenate(y_test_list, axis=1)==b"<=50K", axis=1).reshape(test_array.shape[0], 1)
y_test_array = np.where(y_test_array > K/2, b"<=50K", b">50K")
print("accuracy of test data before pruning: ", Tree.compute_accuracy(y_test_array, test_array[:,0]))
print("number of nodes for each tree before pruning: ", [tree.number_of_nodes(tree.root) for tree in trees])


# start pruning trees and compute accuracy of train, test and validation data for all trees and plot them
for i in range(K):
    train_accuracy_list = []
    test_accuracy_list = []
    validation_accuracy_list = []
    number_of_nodes_list = [trees[i].number_of_nodes(trees[i].root)]
    while(True):
        train_accuracy = trees[i].compute_accuracy(split_train_array_list[i][:,0], trees[i].predict(split_train_array_list[i][:, 1:]))
        test_accuracy = trees[i].compute_accuracy(test_array[:,0], trees[i].predict(test_array[:, 1:]))
        validation_accuracy = trees[i].compute_accuracy(validation_array_list[i][:,0], trees[i].predict(validation_array_list[i][:, 1:]))
        train_accuracy_list.append(train_accuracy)
        test_accuracy_list.append(test_accuracy)
        validation_accuracy_list.append(validation_accuracy)

        trees[i].post_prune(validation_array_list[i][:, 1:], validation_array_list[i][:, 0], how_many=1)

        number_of_nodes_new = trees[i].number_of_nodes(trees[i].root)
        print("number of nodes: ", number_of_nodes_new, end="\r")
        if number_of_nodes_list[-1] == number_of_nodes_new:
            break
        number_of_nodes_list.append(number_of_nodes_new)

    # plot each one of trees
    number_of_nodes_array = np.array(number_of_nodes_list)
    train_accuracy_array = np.array(train_accuracy_list)
    test_accuracy_array = np.array(test_accuracy_list)
    validation_accuracy_array = np.array(validation_accuracy_list)
    plt.figure(i)
    plt.plot(number_of_nodes_array, train_accuracy_list, 'b', number_of_nodes_array, test_accuracy_array, 'r', number_of_nodes_array, validation_accuracy_array, 'k')
    plt.xlabel("number of nodes")
    plt.ylabel("accuracy %")
    plt.title("K = " + str(i+1))
    plt.gca().legend(('train data','test data', 'validation data'))

# compute accuracy on test after pruning
y_test_list = []
for tree in trees:
    y_test_list.append(tree.predict(test_array[:,1:]))
y_test_array = np.sum(np.concatenate(y_test_list, axis=1)==b"<=50K", axis=1).reshape(test_array.shape[0], 1)
y_test_array = np.where(y_test_array > K/2, b"<=50K", b">50K")
print("accuracy of test data after pruning: ", Tree.compute_accuracy(y_test_array, test_array[:,0]))
print("number of nodes for each tree after pruning: ", [tree.number_of_nodes(tree.root) for tree in trees])
    

# show plots
plt.show()