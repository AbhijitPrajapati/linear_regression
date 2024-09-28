import numpy as np
import pandas as pd
from linear_regression import train

def preprocess_data():
    data = pd.read_csv('customer_purchasing_behaviors.csv').drop('user_id', axis=1)
    

    # one hot encoding
    data = pd.concat((data.drop('region', axis=1), pd.get_dummies(data['region'])), axis=1).to_numpy()

    # min max scaling
    scaled_data = np.empty_like(data)

    crange_and_min = []

    for i in range(scaled_data.shape[1]):
        c = data[:, i]
        crange = np.max(c) - np.min(c)
        scaled_data[:, i] = (c - np.min(c)) / crange

        # intermediate values being stored for prediction
        crange_and_min.append((crange, np.min(c)))

    return scaled_data, crange_and_min


if __name__ == '__main__':
    learning_rate = 0.02
    num_epochs = 300

    data, scaler_params = preprocess_data()
    coefs, intercept = train(np.delete(data, 2, axis=1), data[:, 2], learning_rate, num_epochs)
    if int(input('Save? [0/1]: ')):
        np.savez('model', coefs=coefs, intercept=intercept, scaler_params=scaler_params)


    
