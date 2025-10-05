import numpy as np

#step 1 is to get the data
x=np.array([1,2,3,4,5,6])
y=np.array([2,3,8,9,15,20])


#simple equation is y = mx+c
#m is the slope and c is the y intercept

#for theequation get the mean
mean_x=np.mean(x)
mean_Y=np.mean(y)

#execute the equations
m= np.sum((x-mean_x)*(y-mean_Y))/np.sum((x-mean_x)**2)
c= mean_Y-(m*mean_x)

#model equation
print(f"Model: y ={m}*x+{c}")

# value to be predicted is x_test=[2,3,8]
x_test=np.array([2,3,8])

#predicted results for test data
predicted_values= m *x_test+c
print(predicted_values)


# let's do the plots
#import
import matplotlib.pyplot as plt

#Plotting x and y values
plt.scatter(x,y,color='red',marker='o',s=30)  # here s is the marker size

#plotting the decision linear line
# predictions for all x data
y_pred= (m*x)+c
plt.plot(x,y_pred,color='blue')
plt.xlabel('X')
plt.ylabel('Y')
plt.title("Simple Linear Regression")
plt.show()
plt.savefig("Simple Linear Regression Figure 1.jpg")
