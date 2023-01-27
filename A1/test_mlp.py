import numpy as np
import pickle
from nn import read_csv_file, preprocessing_file

STUDENT_NAME = 'VIVEK CHAUHAN, AHALYA SANKARAMAN'
STUDENT_ID = '20955627, 20989393'

def test_mlp(data_file):
	# Load the test set
	# START

	test = read_csv_file(data_file) #loading the data file as csv into list
	for i in range(len(test[0])-1): #as range is indexed from 0 
		preprocessing_file(test, i) # strip and type cast string to float

	#to handle U32 'safe' error due to large matrix multiplication
	test = np.array(test, dtype=np.float32)
    # END

	# Load your network
	# START
	filename_model = r'NeuralNetwork_model.sav'
	saved_nn = pickle.load(open(filename_model, 'rb'))
	# END


	# Predict test set - one-hot encoded
	y_pred = saved_nn.predict(test)

	return y_pred
