#How we will test your code:
from nn import read_csv_file, preprocessing_file
from test_mlp import test_mlp, STUDENT_NAME, STUDENT_ID
from acc_calc import accuracy 
import numpy as np

#just change the file name to be tested
filename = r'train_labels.csv'
test_labels = read_csv_file(filename) #loading the data labels file as csv into list
for i in range(len(test_labels[0])-1):
    preprocessing_file(test_labels, i) 

#just change the file name to be tested
y_pred = test_mlp(r'train_data.csv')

test_labels = np.array(test_labels, dtype=np.float32)

test_accuracy = accuracy(test_labels, y_pred)*100
print(f'Accuracy is : {test_accuracy}')