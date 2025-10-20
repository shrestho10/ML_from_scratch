
import numpy as np

X=np.array([[1,2],[2,3],[3,4],[5,6]],dtype=np.float16)
Y=np.array([100,200,300,400])

ones=np.ones(shape=(len(X),1))

X=np.append(ones,X,axis=1)

coefficients= np.linalg.inv(X.T @ X) @ (X.T @ Y)


def cost(X,y,theta):
    'calculates the cost function'
    predictions=X@theta
    m=len(y)
    cost= 1/m *  np.sum(np.square(predictions-y))

    return cost


def gradient_descent(X,y,theta,alpha,iterations):
    'Iterates and calculates the GD'

    m=len(y)
    cost_history=np.zeros(iterations)
    theta_history=np.zeros((iterations,3))

    for i in range(iterations):
        predictions=X@theta
        theta=theta-alpha*(1/m)*(X.T@(predictions-y))

        cost_history[i]=(cost(X,y,theta))
        theta_history[i,:]=theta.T

        return theta,cost_history,theta_history


theta,cost_history,theta_history=gradient_descent(X,Y,coefficients,0.001,5)
print(cost_history)
print(theta_history)