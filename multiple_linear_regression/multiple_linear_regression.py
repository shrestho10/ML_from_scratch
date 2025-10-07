import numpy as np
#Setting up the datatset

X= np.array([[73,67,43],
             [91,88,64],
             [87,134,58],
             [102,43,37],
             [69,96,70]], dtype='float32')


Y= np.array([56,81,119,22,103],dtype='float32')

#Now we have to impute 1's in X for bias
ones=np.ones(shape=(X.shape[0],1))

#Now impute the ones in X
X= np.append(ones,X, axis=1)

#Now we will calculate the coefficients
beta= np.linalg.inv(X.T.dot(X)).dot((X.T.dot(Y)))

predictions = X.dot(beta)

#Now we will calculate the R2-Score
SSres= np.sum(np.square(Y-predictions))

SStot= np.sum(np.square(Y-np.mean(Y)))

r2score= 1 - (SSres/SStot)

print(r2score)
