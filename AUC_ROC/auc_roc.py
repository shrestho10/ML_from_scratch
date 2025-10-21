import numpy as np

thresholds=[round(i*0.1,1) for i in range(0,11,2)][::-1]
truth_labels = [1, 0, 1, 0, 1]
predicted_probs = [0.9, 0.6, 0.8, 0.2, 0.5]

def compute_aucroc(true,predicted):
    tpr, fpr= [],[]
    for threshold in thresholds:
        TP,FP,TN,FN=0,0,0,0
        for i in range(len(true)):
            if predicted[i]>=threshold:
                if true[i]==1:
                    TP+=1
                else:
                    FP+=1
            else:
                if true[i]==1:
                    FN+=1
                else:
                    TN+=1

        tpr.append(TP/(TP+FN))
        fpr.append(FP/(FP+TN))


    auc_roc= sum(0.5*((fpr[i]-fpr[i-1])*(tpr[i]+tpr[i-1])) for i in range(1,len(tpr)))

    return auc_roc


aucroc_value = compute_aucroc(truth_labels, predicted_probs)
print(f"The AUC-ROC value is: {aucroc_value:.2f}")



