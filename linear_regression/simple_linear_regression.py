import numpy as np

x=np.array([1,2,3,4,5,6])
y=np.array([2,4,6,8,10,12])


#simple equation is y = mx+c
#m is the slope and c is the y intercept
mean_x=np.mean(x)
mean_Y=np.mean(y)


m= np.sum((x-mean_x)*(y-mean_Y))/np.sum((x-mean_x)**2)
c= mean_Y-(m*mean_x)

# value to be predicted is x_test=[2,3,8]
print(f"Model: y ={m}*x+{c}")

x_test=np.array([2,3,8])

predicted_values= m *x_test+c

print(predicted_values)