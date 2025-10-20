

import numpy as np

X=np.array([[1,2],[2,3],[3,4],[5,6]],dtype=np.float16)
Y=np.array([100,200,300,400])



ones=np.ones(shape=(len(X),1))


X=np.append(ones,X,axis=1)
print(X)


coefficients= np.linalg.inv(X.T @ X) @ (X.T @ Y)

predictions = X@ coefficients

print(predictions)