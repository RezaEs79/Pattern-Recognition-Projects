import numpy as np
import pandas as pd
from tree import Tree
import matplotlib.pyplot as plt

# read data
train_df = pd.read_csv("adult.train.10k.discrete")
test_df = pd.read_csv("adult.test.10k.discrete")

# convert data to numpy array
train_array = train_df.to_numpy(dtype=np.string_)
test_array = test_df.to_numpy(dtype=np.string_)

# split train and validation data sets
ratio = 0.75
np.random.shuffle(train_array)
max_index = int(train_array.shape[0] * ratio)
splited_train_array = train_array[:max_index, :]
validation_array = train_array[max_index:, :]

# train tree
my_tree = Tree(splited_train_array[:, 1:], splited_train_array[:, 0])
my_tree.fit()

# start pruning tree and compute accuracy of train, test and validation data
train_accuracy_list = []
test_accuracy_list = []
validation_accuracy_list = []
number_of_nodes_list = [my_tree.number_of_nodes(my_tree.root)]
while(True):
    train_accuracy = my_tree.compute_accuracy(splited_train_array[:,0], my_tree.predict(splited_train_array[:, 1:]))
    test_accuracy = my_tree.compute_accuracy(test_array[:,0], my_tree.predict(test_array[:, 1:]))
    validation_accuracy = my_tree.compute_accuracy(validation_array[:,0], my_tree.predict(validation_array[:, 1:]))
    train_accuracy_list.append(train_accuracy)
    test_accuracy_list.append(test_accuracy)
    validation_accuracy_list.append(validation_accuracy)

    my_tree.post_prune(validation_array[:, 1:], validation_array[:, 0], how_many=1)

    number_of_nodes_new = my_tree.number_of_nodes(my_tree.root)
    print("number of nodes: ", number_of_nodes_new, end="\r")
    if number_of_nodes_list[-1] == number_of_nodes_new:
        break
    number_of_nodes_list.append(number_of_nodes_new)

# plot accuracy of train, test and validation data based on number of nodes
number_of_nodes_array = np.array(number_of_nodes_list)
train_accuracy_array = np.array(train_accuracy_list)
test_accuracy_array = np.array(test_accuracy_list)
validation_accuracy_array = np.array(validation_accuracy_list)
print(number_of_nodes_array)

plt.plot(number_of_nodes_array, train_accuracy_list, 'b', number_of_nodes_array, test_accuracy_array, 'r', number_of_nodes_array, validation_accuracy_array, 'k')
plt.xlabel("number of nodes")
plt.ylabel("accuracy %")
plt.gca().legend(('train data','test data', 'validation data'))
plt.show()