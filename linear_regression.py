import numpy as np

MSE = lambda target, pred: np.sum((target - pred) ** 2) / len(target)
DMSE = lambda target, pred: 2 / len(target) * (pred - target)

FOREWARD = lambda coefs, intercept, data: np.sum(coefs * data, axis=1) + intercept

def train(features, target, learning_rate, num_epochs, verbose=1):
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

        if epoch % 10 == 0 and verbose:
            # compute loss
            loss = MSE(target, FOREWARD(coefs, intercept, features))
            print(f'Epoch: {epoch}\nLoss: {loss}')
    
    return coefs, intercept