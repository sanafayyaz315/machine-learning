'''

STEPS:
1. create data: x1, x2, y, m=200, n=2
2. add bias term x0 to vector X
3. normalize every feature of : [x - min(x)] / [max(x) - min(x)] where x = x0, x1, x2
4. initialize params (W=w0,w1,w2), W.shape=(3,)(2 for x1 and x2, 1 for the bias term)
5. sigmoid function
6. calculate y_hat or a: y_hat = sigmoid(W*X + b),  y_hat.shape=(200,)
7. calculate loss function: binary cross-entropy
8. calculate derivitives 
9. gradient descent
10. train function

'''
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

def create_data(m,n):
    X = np.random.rand(m,n)
    examples = m//2
    y = np.concatenate((np.zeros(examples), np.ones(examples)), axis=0)
    return X, y

def read_data(path, type):
    if type == 'csv':
        data = pd.read_csv(path)
    if type == 'excel':
        data = pd.read_excel(path)
    X = data[['gre', 'gpa', 'rank']].values
    y = data[['admit']].values
    return X,y

def add_bias_feature(X, m):
    x0 = np.ones((m,1))
    X = np.concatenate((x0, X), axis=1)
    return X

def normalize(X, mode):
    min = np.min(X, axis=0)
    max = np.max(X, axis=0)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    if mode == 'min_max':
        X = (X - min)/(max - min)
    
    if mode == 'z_score':
        X = (X - mean)/std
    return X

def init_params(n):
    w = np.random.rand(n+1)
    return w

def sigmoid(z):
    z = 1/(1+np.exp(-z))
    return z

def forward(X, w, threshold_val=0.7):
    threshold_val = threshold_val
    z = np.dot(X,w)
    a = sigmoid(z)
    a = np.where(a >= threshold_val, 1, 0)
    return a

def cross_entropy(y, a, m):
    epsilon = 1e-15  # small constant to avoid division by zero
    a = np.clip(a, epsilon, 1 - epsilon)  # clip predicted probabilities to avoid log(0) and log(1)
    loss = -(y*np.log(a) + (1-y)*np.log(1-a))
    # loss = np.sum(loss)
    loss = np.mean(loss)

    return loss   

def gradients(X, y, a, w, m):
    dw = []
    for i in range(X.shape[1]):
        gradient = (np.sum((a - y) * X[:,i])) / m
        dw.append(gradient)
    dw = np.array(dw)
    return dw

def backward(w, dw, lr):
    w_updated = w - lr*dw
    return w_updated

def train(X, y, w, m, lr=0.001, epochs=100):
    for i in range(epochs):
        a = forward(X, w)
        loss = cross_entropy(y, a, m)
        dw = gradients(X, y, a, w, m)
        w = backward(w, dw, lr=lr)
        if i%1000 == 0:
            print('at epoch', i ,'loss =', loss, 'w_updated =', w)
    return loss, w



if __name__ == "__main__":
    path = './logistic regression/student_data.csv'
    X, y = read_data(path, type='csv')
    m = X.shape[0]
    n = X.shape[1]
    epochs = 10000
    # X, y = create_data(m,n)
    X = add_bias_feature(X, m)
    X = normalize(X, mode='std')
    w = init_params(n)
    loss, w_updated = train(X, y, w, m, lr=0.009, epochs=epochs)
    # plt.plot(epochs, loss)
    # plt.show()
    print ()

    
    



