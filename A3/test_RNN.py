import pandas as pd
import numpy as np
#import time
from matplotlib import pyplot as plt
#import os
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


if __name__ == "__main__":
    #loading model
    model = load_model("models/20955627_RNN_model.model")
    test_data = pd.read_csv(r'data/test_data_RNN.csv')

    n_cols = len(test_data.columns)
    cols = ["feature_"+str(i) for i in range(n_cols-2)]
    x_test = test_data[cols].values
    y_test = test_data['target_open'].values.reshape((-1, 1))
    dates = test_data['date'].values.reshape((-1, 1))

    minmax_x = MinMaxScaler()
    x_test_scaled = minmax_x.fit_transform(x_test)
    minmax_y = MinMaxScaler()
    y_test_scaled = minmax_y.fit_transform(y_test)
    x_test_scaled = np.reshape(x_test_scaled, (x_test_scaled.shape[0], x_test_scaled.shape[1], 1))

    y_pred = model.predict(x_test_scaled)
    y_pred_inverse =  minmax_y.inverse_transform(y_pred)
    print("RMSE Value is = ", np.sqrt(np.mean(((y_pred_inverse - y_test)**2))))
    columns = ['true', 'predicted', 'date']
    final = pd.DataFrame(data=np.concatenate((y_test, y_pred_inverse, dates), axis=1), columns=columns)
    final['date'] = pd.to_datetime(final['date'], format="%m/%d/%y")
    final = final.sort_values(by='date')
    plt.figure()
    plt.plot(final['date'], final['true'])
    plt.plot(final['date'], final['predicted'])
    plt.xlabel('Date')
    plt.ylabel('Open')
    plt.legend(['True', 'Predicted'])
    plt.show()