import numpy as np

class Node:
    def __init__(self, feature_index=None, children=None, parent=None, leaf=False, leaf_value=None, y=None):
        self.children = children
        self.feature_index = feature_index
        self.parent = parent
        self.leaf = leaf
        self.leaf_value = leaf_value
        self.depth = 0
        self.validation_error = 0

class Tree:
    def __init__(self, x, y):
        """
            train_set must be a numpy array with size n * m where n is number of patterns and m is number of features
            first column of train_set should be the feature we want to predict
        """
        self.train_x = x
        self.train_y = y
        self.root = Node(children={})

    def fit(self):
        self.build_tree(self.root, self.train_x, self.train_y, [], 0)

    def build_tree(self, node, x, y, used_features, depth):
        # assign current node depth (this will be used in post pruning)
        node.depth = depth

        # assign leaf value of current node assuming it is a leaf (this will be used in post pruning)
        vals, count = np.unique(y, return_counts=True)
        mode_index = np.argmax(count)
        node.leaf_value = vals[mode_index]

        # recusive function stop conditionsS
        if np.unique(y).size == 1 or len(used_features) == x.shape[1]:
            node.leaf = True
            return

        # compute IG for all columns of x and choose the maximum feature
        n_cols = x.shape[1]
        max_ig = -1
        max_ig_index = 0
        for i in range(n_cols):
            if i not in used_features:
                ig = self.information_gain(y, x[:, i])
                if ig > max_ig:
                    max_ig = ig
                    max_ig_index = i

        # assign best feature as root of current position of tree
        used_features.append(max_ig_index)
        node.feature_index = max_ig_index
        # build next nodes of tree
        for unique_value in np.unique(x[:, max_ig_index]):
            child_node = Node(parent=node, children={})
            node.children[unique_value] = child_node
            # split data and continue building tree recutrsively
            child_data_indexes = x[:, max_ig_index] == unique_value
            self.build_tree(child_node, x[child_data_indexes, :], y[child_data_indexes], used_features[:], depth=depth+1)

    def predict(self, x):
        # check if we have a tree
        if not self.root.leaf and not self.root.children:
            print("there is no tree here!!")
            return

        # make an arbitary y as output of tree
        y = np.full(shape=(x.shape[0], 1), fill_value=max(self.train_y, key=len))

        # start predicting using a recurent function
        self.go_down(self.root,x ,np.any(x == x, axis=1), y)

        # return result
        return y

    # helper function for predict
    def go_down(self, node, x, indexes, y):
        # stop if current node is a leaf
        if node.leaf:
            y[indexes] = node.leaf_value
            return

        feature_index = node.feature_index
        for child_label in node.children:
            new_indexes = np.logical_and(x[:, feature_index] == child_label, indexes)
            self.go_down(node.children[child_label], x, new_indexes, y)

    def post_prune(self, val_x, val_y, how_many=1):
        # compute validation error for all nodes
        self.compute_validation_error(self.root, val_x, val_y)

        # find all nodes that are not leaves
        nodes_not_leaves = self.find_nodes_not_leaves(self.root)

        # sort nodes based on depth
        nodes_not_leaves.sort(key= lambda x: x.depth, reverse=True)
        
        # loop through all nodes that aren't leaf and check if their validation error is less than total of their children
        counter = 0
        for node in nodes_not_leaves:
            node_validation_error = node.validation_error
            children_validation_error = 0
            # compute sum of validation error of leaves that are children of current node
            sum_of_leaf_children_validation_error = self.sum_of_leaf_children_validation_error(node)
            if node_validation_error < sum_of_leaf_children_validation_error:
                counter += 1
                node.leaf = True
                node.children = {}
            if counter == how_many:
                break

    def compute_validation_error(self, node, val_x, val_y):
        node.validation_error = val_x.shape[0] - np.sum(val_y == node.leaf_value) 

        for child_label in node.children:
            indexes = val_x[:, node.feature_index] == child_label
            self.compute_validation_error(node.children[child_label], val_x[indexes, :], val_y[indexes])

        
    def find_nodes_not_leaves(self, node):
        if node.leaf:
            return []
        
        nodes = [node]
        for child in node.children.values():
            nodes += self.find_nodes_not_leaves(child)

        return nodes

    def sum_of_leaf_children_validation_error(self, node):
        if node.leaf:
            return node.validation_error
        
        s = 0
        for child in node.children.values():
            s += self.sum_of_leaf_children_validation_error(child)
        
        return s

    def number_of_nodes(self, node):
        n_nodes = 1
        for child in node.children.values():
            n_nodes += self.number_of_nodes(child)
        return n_nodes

    @staticmethod
    def compute_accuracy(y1, y2):
        y2 = y2.reshape(y1.shape)
        return np.sum(y1 == y2) / np.max(y1.shape) * 100

    @staticmethod
    def entropy(v, base=2):
        n_labels = len(v)

        if n_labels <= 1:
            return 0

        values, counts = np.unique(v, return_counts=True)
        probs = counts / n_labels
        
        return np.sum(-probs * (np.log(probs) / np.log(base)))
    
    @classmethod
    def information_gain(cls, class_array, feature_array):
        """
            computes IG(class, feature)
            class_array, feature_array both are n*1 size vectors
        """
        total_size = np.max(class_array.shape)
        IG = Tree.entropy(class_array)
        for unique_value in np.unique(feature_array):
            unique_value_indexes = feature_array == unique_value
            IG -= Tree.entropy(class_array[unique_value_indexes]) * np.sum(unique_value_indexes) / total_size
        return IG

if __name__ == "__main__":
    test_data = [['Yes', 'Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'],
                 ['Yes', 'Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same'],
                 ['No', 'Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Same'],
                 ['Yes', 'Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change'],
                 ['No', 'Sunny', 'Warm', 'Normal', 'Weak', 'Warm', 'Same']]
    train_set = np.array(test_data, dtype=np.string_)
    my_tree = Tree(train_set[:, 1:], train_set[:, 0])
    my_tree.fit()
    print(my_tree.predict(train_set[:,1:]))

    test_data2 = [["plastic", "big", "black"],
                  ["plastic", "big", "black"], 
                  ["plastic", "big", "black"], 
                  ["plastic", "small", "white"], 
                  ["wood", "small", "white"],
                  ["wood", "small", "white"], 
                  ["wood", "small", "white"], 
                  ["wood", "big", "white"]]
    train_set2 = np.array(test_data2, np.string_)
    my_tree2 = Tree(train_set2[:, 1:], train_set2[:, 0])
    my_tree2.fit()
    print(my_tree2.predict(train_set2[:,1:]))
    print(Tree.compute_accuracy(my_tree2.predict(train_set2[:,1:]), train_set2[:,0]))

    my_tree2.post_prune(train_set2[:,1:], train_set2[:,0])
    print(my_tree2.number_of_nodes(my_tree2.root))
