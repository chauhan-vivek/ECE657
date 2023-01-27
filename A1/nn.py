#Importing Libraries
import time #for time taken in every iteration
from csv import reader #to read csv files
from math import exp #to use in sigmoid and softmax function
import pickle #to save the network to file

import numpy as np #used for array numerical calculations
import pandas as pd #for handling dataframe
from sklearn.model_selection import train_test_split # for train_test_split

#for plotting
#from matplotlib import pyplot as plt
#import matplotlib
#%matplotlib inline

class NeuralNetwork():
    """
    class consists of methods for forward pass , backpropagation , error calculation, activation function(sigmoid) 
    and other required methods. We pass dfault hyperparameters of our choice to initialize the methods. 
    """
    def __init__(self, shape_list, ep=10, learning_rate=0.1):
        """
        method used to call initialize the dimensions of matrixes and hyperparamaters
        """
        self.shapes = shape_list #specifices the input vector size, number of neurons in hidden layer & output classes  
        self.epoch = ep #number of epochs to train on
        self.learning_rate = learning_rate # learning rate of 0.1 worked better 

        #call to method to store network parameters in a dictionary
        self.parameters = self.network_initialization()

    def sigmoid(self, x, d=False):
        """
        it accepts an array and a parameter to calculate derivate
        returns derivative or normal sigmoid/logistic function
        """
        if d: #derivative is True, used in backpropagation
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x)) #sigmoid function 

    def softmax(self, x, d=False):
        """
        it accepts an array and a parameter to calculate derivate
        returns derivative or normal softmax function which is exp(x) divided by summation of exp(x) 
        for entire row       
        """
        exps = np.exp(x - x.max())
        if d:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def network_initialization(self):
        """
        initializes the dimensions of matrices in input layer, hidden layer and output layer
        returns the parameters as dictionary where key represesnts the parameter referenced
        """
        ip_l=self.shapes[0]
        hd_1=self.shapes[1]
        op_l=self.shapes[2]

        parameters = {
            'w1':np.random.randn(hd_1, ip_l) * np.sqrt(1. / hd_1), # 128*784 : weights(synaptics) between ip layer and hidden layer
            'b1':np.random.randn(hd_1), #128 bias units
            'w2':np.random.randn(op_l, hd_1) * np.sqrt(1. / op_l), #4*128 : weights(synaptics) between hidden layer and output layer
            'b2':np.random.randn(op_l) #4 bias units
        }

        return parameters

    def forwardpass(self, x_train):
        """
        calculates the multiplication of weight matrix with input vector ans passed through an
        activation function , it accepts only the training data and returns the predicted class probabilities
        """
        parameters = self.parameters # initialized in network_initialisation method

        # input vector
        parameters['a0'] = x_train

        # from input layer to hidden layer
        parameters['z1'] = np.dot(parameters["w1"], parameters['a0']) + parameters['b1'] #adding bias 
        parameters['a1'] = self.sigmoid(parameters['z1']) #sigmoid activation function

        # from hidden layer to output layer
        parameters['z2'] = np.dot(parameters["w2"], parameters['a1']) + parameters['b2'] #adding bias
        parameters['a2'] = self.softmax(parameters['z2']) #softmax activation function

        return parameters['a2']

    def backpass(self, y_train, output):
        '''
            implememts backpropagation algorithm for calculating the delta in weights
            using chaining method to update parameters.
        '''
        parameters = self.parameters
        #delta in weight in each layer
        delta_w = {}
        output_error = output - y_train # error in output
        # Calculate the W2 update
        error = 2 * output_error / output.shape[0] * self.softmax(parameters['z2'], d=True) #default derivative parameter is set to False
        delta_w['w2'] = np.outer(error, parameters['a1'])

        # Calculate W1 update
        error = np.dot(parameters['w2'].T, error) * self.sigmoid(parameters['z1'], d=True) #default derivative parameter is set to False
        delta_w['w1'] = np.outer(error, parameters['a0'])

        return delta_w

    def update(self, delta_in_w):
        '''
            initial weight is changed based on the learning rate and delta calculated in weight
            in backpass : w(new) = w(old) - learning_rate*delta_in_w
        '''
        
        for key, value in delta_in_w.items(): #using the dictionary items to refernce the parameter to be updated
            self.parameters[key] -= self.learning_rate * value 
            
    def predict(self, x_val):
        """"
        accepts the intsance and an array 
        returns one hot encoded label predictions
        """
        prediction = []
        
        for x in x_val:
            y_prob = self.forwardpass(x)
            pred = np.argmax(y_prob) #give the index with highest probability in row
            #one_hot = pd.get_dummies(pred)
            prediction.append(pred)
        
        return pd.get_dummies(prediction).values #returns one hot encoded df and then numpy array

    def accuracy(self, x_val, y_val):
        '''
        accepts data array and labels array
        calculates accuracy by using the predicted label from forward pass compared with true label provide as input
        return accuracy for the dataset
        '''
        predictions = []
        
        #using the validation dataset for calculating accuracy
        for x, y in zip(x_val, y_val):
            y_prob = self.forwardpass(x)
            pred = np.argmax(y_prob)
            predictions.append(pred == np.argmax(y))
        
        return np.mean(predictions)
    
    #Method to call to train the network
    def training(self, x_train, y_train, x_val, y_val):
        
        """
        method called to train neural network with training data arrays and validation data arrays 
        returns to console accuracy with every iteration  and time till that iteration
        """
        start_time = time.time()
        for i in range(self.epoch):
            for x,y in zip(x_train, y_train):
                output = self.forwardpass(x)
                delta_in_w = self.backpass(y, output)
                self.update(delta_in_w)
            
            accuracy = self.accuracy(x_val, y_val)
            #y_pred = self.predict(x_val)
            #print(f'{y_pred}')
            #print('Epoch: {0}, Total time spent : {1:.2f}s, Accuracy: {2:.2f}%'.format(
            #    i+1, time.time() - start_time, accuracy * 100
            #))

def read_csv_file(file):
    """
    This function accepts the filename agrument which is the csv file that is to be read
    and outputs the dataset
    """
    df = [] #empty list to store the dataset row wise
    with open(file, 'r') as f: #file opened in read mode only
        file_read = reader(f)
        for row in file_read:
            if not row:
                #if the file has reached the end or the row is blank
                continue
            df.append(row)
    return df

# Type casts string column to float as normalized pixel values are expected to be float
def preprocessing_file(df, col):
    """
    This function accepts the dataset returned by load_csv function and column as argument to strip
    any blank space at the end of each cell and casts it into float
    """
    for row in df:
        row[col] = float(row[col].strip())


filename = r'train_data.csv'
x = read_csv_file(filename) #loading the data file as csv into list
for i in range(len(x[0])): #as range is indexed from 0 
    preprocessing_file(x, i) # strip and type cast string to float
    
filename_2 = r'train_labels.csv'
y = read_csv_file(filename_2) #loading the data labels file as csv into list
for i in range(len(y[0])):
    preprocessing_file(y, i) 

#to handle 'U32 safe' error 
x = np.array(x, dtype=np.float32) 
y = np.array(y, dtype=np.float32)

print(f'Length of train_data is {len(x)} train_labels is {len(y)}')
#using the train_test_split function from scikit learn to split our dataset into training and validation
#data set in the ratio of 80:20 
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.20, random_state=42)
#print(f'Length of Training dataset is : {len(x_train)} , {len(y_train)}')
#print(f'Length of Training dataset is : {len(x_val)} , {len(y_val)}')

#initilalizing the neural network with our data shapes and calling the training method
nn = NeuralNetwork(shape_list=[784, 128, 4])
nn.training(x_train, y_train, x_val, y_val)

# save the model to local file
filename = 'NeuralNetwork_model.sav'
pickle.dump(nn, open(filename, 'wb'))


#########################
#to plot learning rate vs accuracy plot
'''
learning_rate_list = [0.001, 0.01, 0.1, 1]
acc_list = []
for l in learning_rate_list:
    nn = NeuralNetwork(shape_list=[784, 128, 4], learning_rate = l, ep = 10)
    nn.training(x_train, y_train, x_val, y_val)
    acc= nn.accuracy(x_val, y_val)
    acc_list.append(acc)
print(acc_list)

x = learning_rate_list
y = acc_list 
plt.plot(x,y) #line plot
plt.xlabel('Learning Rate') 
plt.ylabel('Accuracy')
plt.title("Learning rate vs Accuracy plot")
'''