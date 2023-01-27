import pandas as pd
import numpy as np
#import time
from matplotlib import pyplot as plt
import os
#from pickle import dump

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


def prepare_dataset(file):
    """
    This function creates dataset using three consecutive dates data and writes into 
    seperate csv files for test and train
    
    Arguments:
        file : file path to be read
    
    """
    raw_df = pd.read_csv(file) # reading csv
    target_features = [' Open', ' High', ' Low',' Volume'] #columns of interest
    dates = raw_df.Date.values #as array
    features = raw_df[target_features].values # as array
    
    X, y = [],[]
    
    for i in range(len(features)-3):
        #selecting three dates date as features and current date close value as label
        x_i = features[i+1:i+4]
        y_i = features[i][0]
        #dates are either mm/dd/yy or mm/dd/yyyy format so converting all mm/dd/yy format
        date = dates[i] if len(dates[i]) == 8 else (dates[i][:6] + dates[i][8:]) #mm/dd/yy
        X.append(x_i.reshape(12))
        y.append([y_i, date])
    
    X = np.array(X)
    y = np.array(y)
    
    #shuffling data
    n_records = len(X)
    idxs = np.random.permutation(n_records)
    X = X[idxs]
    y = y[idxs]

    #splitiing data into train and test
    training_size = 0.7
    split_idx = int(n_records*training_size)
    x_train, x_test, y_train, y_test = X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
    
    data_cols = ["feature_"+str(i) for i in range(12)]
    data_cols += ["target_open", 'date']

    train = np.concatenate((x_train, y_train), axis = 1)
    test = np.concatenate((x_test, y_test), axis = 1)
    
    #creating train and test dataframes
    df_train = pd.DataFrame(train, columns = data_cols)
    df_test = pd.DataFrame(test, columns = data_cols)
    
    #writing to train and test csv files
    df_train.to_csv(r'data/train_data_RNN.csv', index = False)
    df_test.to_csv(r'data/test_data_RNN.csv', index = False)

if __name__ == "__main__": 
    #prepare_dataset(r'data/q2_dataset.csv')
    train_data = pd.read_csv(r'data/train_data_RNN.csv')
    n_cols = len(train_data.columns)
    cols = ["feature_"+str(i) for i in range(n_cols-2)]
    x_train = train_data[cols].values
    y_train = train_data['target_open'].values.reshape((-1, 1))
    dates = train_data['date'].values.reshape((-1, 1))
    
    #using minmax scaler to scale the data
    minmax_x = MinMaxScaler()
    x_train_scaled = minmax_x.fit_transform(x_train)
    minmax_y = MinMaxScaler()
    y_train_scaled = minmax_y.fit_transform(y_train)
    
    x_train_scaled = np.reshape(x_train_scaled, (x_train_scaled.shape[0], x_train_scaled.shape[1], 1))
    
    BATCH_SIZE = 1
    EPOCHS = 5
    
    #creating our LSTM model with two LSTM layers with different number of units 
    #and using Adam optimizer to evaluate loss on MSE criterion
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape= (x_train_scaled.shape[1], 1)))
    #model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2)) # helps in generalization, reduces overfitting
    model.add(Dense(50)) # fully connected layer
    model.add(Dense(1)) #output layer
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    history = model.fit(x_train_scaled, y_train_scaled, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose =True)
    
    model.save("models/20955627_RNN_model.model")
    print("Tranining Loss is :", model.evaluate(x_train_scaled, y_train_scaled))
    
    #prediction on x_train
    y_pred = model.predict(x_train_scaled)
    y_pred_inverse =  minmax_y.inverse_transform(y_pred)
    columns = ['true', 'predicted', 'date']
    final = pd.DataFrame(data=np.concatenate((y_train, y_pred_inverse, dates), axis=1), columns=columns)
    final['date'] = pd.to_datetime(final['date'], format="%m/%d/%y")
    final = final.sort_values(by='date')
    
    #visualizing the deviation between true and predicted values
    plt.figure()
    plt.plot(final['date'], final['true'])
    plt.plot(final['date'], final['predicted'])
    plt.xlabel('Date')
    plt.ylabel('Open')
    plt.legend(['True', 'Predicted'])
    plt.show()
    
    #plotiing Loss
    plt.figure()
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.show()