import numpy as np
import csv
import sys
import pickle
from validate import validate
import train 
"""
Predicts the target values for data in the file at 'test_X_file_path', using the model learned during training.
Writes the predicted values to the file named "predicted_test_Y_de.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def import_data_and_model(test_X_file_path, model_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    model = train.get_model()
    return test_X, model

def predict_for(X, root):
    current_node = root
    while True:
        
        if current_node.left is None and current_node.right is None:
            return current_node.predicted_class
        
        test_feature_value = X[current_node.feature_index]
        
        if test_feature_value >= current_node.threshold:
            current_node = current_node.right
        else:
            current_node = current_node.left


def predict_target_values(test_X, model):
    pred_Y=[]
    for X in test_X:
        pred_Y.append(predict_for(X,model))
    return np.array(pred_Y)
    
def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X, model = import_data_and_model(test_X_file_path, 'MODEL_FILE.sav')
    pred_Y = predict_target_values(test_X, model)
    write_to_csv_file(pred_Y, "predicted_test_Y_de.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    # validate(test_X_file_path, actual_test_Y_file_path="train_Y_de.csv") 
