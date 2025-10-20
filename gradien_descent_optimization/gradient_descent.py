
import numpy as np

def cost_function(X,y,theta):
    'returns the cost from predictions'

    m=len(y)
    predicitons= X.dot(theta)
    cost= (1/m)*np.sum(np.square(predicitons-y))
    return cost



def gradient_descent(X,y,theta,alpha,iterations):

    m=len(y)
    cost_history=np.zeros(iterations)
    theta_history=np.zeros((iterations,2))

    for i in range(iterations):
        prediction= X.dot(theta)
        theta= theta - (1/m) * alpha*(X.T.dot(prediction-y))
        theta_history[i,:]=theta.T
        cost_history[i]=cost_function(X,y,theta)

    return theta, cost_history, theta_history



X= 2* np.random.rand(100,1)
y=4+3*X+ np.random.rand(100,1)

lr=0.01
n_iter=5

theta=np.random.randn(2,1)

print(X)

X_b=np.c_[np.ones((len(X),1)),X]

theta, cost_history, theta_history = gradient_descent(X_b,  y, theta, lr, n_iter)


print(cost_history)






