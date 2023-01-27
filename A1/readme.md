testing_function can be directly utilized to test the assignment

Data and label arrays are converted numpy 'float32' array

Note : Please convert to numpy array float32 befor training the dataset
#to handle 'U32 safe' error
#x = np.array(x, dtype=np.float32) 

Comment out the last part which plots accuracy vs learning rate as it runs 15 epochs of training for 4 different learning rates. 