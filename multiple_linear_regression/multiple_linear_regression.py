import numpy as np
#Setting up the datatset

X= np.array([[73,67,43],
             [91,88,64],
             [87,134,58],
             [102,43,37],
             [69,96,70]], dtype='float32')


Y= np.array([56,81,119,22,103],dtype='float32')


#but for the bias term we will 1 in the features of X

ones=np.ones(shape=(len(X),1))  # so the shape is (5,1)

X=np.append(ones,X,axis=1)  # added in column

#Now lets take the co-efficients
beta=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)


predictions = X.dot(beta)

residuals = np.sum(np.square(predictions-Y))
total= np.sum(np.square(Y-np.mean(Y)))
r2_score= 1 - (residuals/total)

print(r2_score)
