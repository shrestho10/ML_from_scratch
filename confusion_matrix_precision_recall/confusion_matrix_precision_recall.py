
import numpy as np

y_true=np.array([1,0,0,0,1,1,1,0,1,0,0,0,1])
y_pred=np.array([1,0,1,0,1,1,1,1,0,0,0,0,1])


#Now let's calculate the Confusion Matrix Essentials

TP= np.sum((y_pred==1) & (y_true==1))

TN= np.sum((y_pred==0) & (y_true==0))

FP= np.sum((y_pred==1) & (y_true==0))

FN= np.sum((y_pred==0) & (y_true==1))




def precision(TP,FP):
    return TP/(TP+FP)

def recall(TP,FN):
    return TP/(TP+FN)

def accuracy(TP,FP,TN,FN):
    return (TP+TN)/(TP+FN+FP+TN)


print("Confusion Matrix")

print(f"TP: {TP}, FP: {FP}\n TN:{TN}, FN {FN}")

print(f"Precision: {precision(TP,FP)}")

print(f"Recall: {recall(TP,FN)}")

print(f"Accuracy: {accuracy(TP,FP,TN,FN)}")