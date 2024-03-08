from src import *
# Main function
def main():
    print("################################ Part A ###########################")
    k_values = [1, 2, 3]
    print("The kNN classifier with k=1, 2, 3 and leave-one-out method for ' Iris dataset ' ")
    X, y = load_iris_dataset()
    for k in k_values:
        correct_predictions = leave_one_out_cross_validation(X, y, [k])
        accuracy = correct_predictions[k] / len(X)
        print(f"k={k}, Accuracy: % {accuracy *100:.2f}")
    print("")

    print("The kNN classifier with k=1, 2, 3 and leave-one-out method for ' Liquid dataset ' ")
    X, y = load_liquid_dataset()
    for k in k_values:
        correct_predictions = leave_one_out_cross_validation(X, y, [k])
        accuracy = correct_predictions[k] / len(X)
        print(f"k={k}, Accuracy: % {accuracy *100:.2f}")
    print("")

    print("The kNN classifier with k=1, 2, 3 and leave-one-out method for ' Normal dataset ' ")
    X, y = load_normal_dataset('train')
    for k in k_values:
        correct_predictions = leave_one_out_cross_validation(X, y, [k])
        accuracy = correct_predictions[k] / len(X)
        print(f"k={k}, Accuracy: % {accuracy *100:.2f}")
    print("\n")
    print("################################ Part B ###########################")
    X, y = load_iris_dataset()
    correct_predictions = leave_one_out_cross_validation_MMD(X, y)
    accuracy = correct_predictions / len(X)
    print(f"MMD Classifier with Leave-One-Out for ' Iris dataset ' , Accuracy: % {accuracy*100:.2f}")

    X, y = load_liquid_dataset()
    correct_predictions = leave_one_out_cross_validation_MMD(X, y)
    accuracy = correct_predictions / len(X)
    print(f"MMD Classifier with Leave-One-Out for ' Liquid dataset ' , Accuracy: % {accuracy*100:.2f}")

    X_tr, y_tr = load_normal_dataset('train')
    X_ts, y_ts = load_normal_dataset('test')
    correct_predictions = Test_dataset_validation(X_tr, y_tr, X_ts, y_ts)
    accuracy = correct_predictions / len(X_ts)
    print(f"MMD Classifier with Testing on test dataset for ' Normal dataset ' , Accuracy: % {accuracy*100:.2f}")
 
 
if __name__ == "__main__":
    main()
