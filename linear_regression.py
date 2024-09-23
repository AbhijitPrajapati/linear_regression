import numpy as np

learning_rate = 0.02
num_epochs = 300

MSE = lambda target, pred: np.sum((target - pred) ** 2) / len(target)
DMSE = lambda target, pred: 2 / len(target) * (pred - target)

FOREWARD = lambda coefs, intercept, data: np.sum(coefs * data, axis=1) + intercept

def train(features, target):
    if len(features.shape) == 1:
        raise ValueError('Feature arrays must be 2D')

    coefs = np.zeros(features.shape[1])
    intercept = 0

    for epoch in range(1, num_epochs + 1):
        pred = FOREWARD(coefs, intercept, features)

        # dot product of gradient of loss func. w.r.t preds
        # and gradient of preds w.r.t coefs (input data)
        dloss = DMSE(target, pred)
        coef_grads, intercept_grad = np.dot(dloss, features), np.average(dloss)
        # adjust parameters
        coefs = coefs - learning_rate * coef_grads
        intercept = intercept - learning_rate * intercept_grad

        if epoch % 10 == 0:
            # compute loss
            loss = MSE(target, FOREWARD(coefs, intercept, features))
            print(f'Epoch: {epoch}\nLoss: {loss}')
    
    return coefs, intercept

def predict(input, target_ind, coefs, intercept, scaler_params=None):
    if not scaler_params:
        # establishes scaling parameters that do not do anything
        scaler_params = [(1, 0) for _ in range(len(input) + 1)]

    scaled_input = [(i - m) / r for i, (r, m) in zip(input, np.delete(scaler_params, target_ind, axis=0))]
    scaled_output = FOREWARD(coefs, intercept, np.array(scaled_input).reshape(1, -1))[0]
    unscaled_output = scaled_output * scaler_params[target_ind][0] + scaler_params[target_ind][1]

    return unscaled_output