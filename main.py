import numpy as np
import pandas as pd
from linear_regression import train, predict

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
    # data, scaler_params = preprocess_data()
    # coefs, intercept = train(np.delete(data, 2, axis=1), data[:, 2])

    # if int(input('Save? [0/1]: ')):
    #     np.savez('model', coefs=coefs, intercept=intercept, scaler_params=scaler_params)
    
    # print(predict(np.array([25,45000,4.5,12, 1, 0, 0, 0]), 2, coefs, intercept, scaler_params))

    model = np.load('model.npz', allow_pickle=True)
    coefs, intercept, scaler_params = model['coefs'], model['intercept'], model['scaler_params']
    model.close()

    print(predict(np.array([25,45000,4.5,12, 1, 0, 0, 0]), 2, coefs, intercept, scaler_params))


    
