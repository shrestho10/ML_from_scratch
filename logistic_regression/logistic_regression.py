import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def cost_function(h,y):
    'here y is the label and h is the prediction'
    return ((-y *np.log(h))-((1-y)*(np.log(1-h)))).mean()


def logistic_regression(X,y,iterations,learning_rate):
    intercept=np.ones((X.shape[0],1))
    X= np.concatenate((intercept,X),axis=1)

    #let's intialize weights

    theta= np.zeros(X.shape[1])

    for i in range(iterations):
        z=X @theta
        h=sigmoid(z)

        gradient= (X.T @ (h-y))/y.size

        theta -= learning_rate*gradient

        z=X@theta
        h=sigmoid(z)
        loss= cost_function(h,y)

        if i%1000 ==0:
            print(f"Loss after iteration {i} : {loss}")

    return theta




def predict_proba(X,theta):
    intercept=np.ones((X.shape[0],1))
    X= np.concatenate((intercept,X),axis=1)

    return sigmoid(X@theta)

def predict(X, theta, threshold=0.5):
    return predict_proba(X,theta)>=threshold

np.random.seed(0)
num_observations = 1000
x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)
X = np.vstack((x1, x2)).astype(np.float32)
y = np.hstack((np.zeros(num_observations), np.ones(num_observations)))

# Split into train and test
train_ratio = 0.7
idx = int(X.shape[0] * train_ratio)
X_train, X_test = X[:idx, :], X[idx:, :]
y_train, y_test = y[:idx], y[idx:]

# Training
theta = logistic_regression(X_train, y_train, iterations = 30000, learning_rate = 0.1)

# Prediction and Accuracy calculation
y_pred = predict(X_test, theta)
accuracy = (y_pred == y_test).mean()
print('Accuracy:', accuracy)