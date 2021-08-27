import pickle
import os
import numpy as np
import random
import tabulate
from tabulate import tabulate

def predictValueWithThreshold(threshold, DLOutputs, RealLabel):
    pred_value = []
    metricType = []
    samples = len(DLOutputs)
    for i in range (samples):
        if DLOutputs[i] >= threshold:#0.5
            pred_value.append(1)
            if RealLabel[i] == 1:
                metricType.append("TP")
            else:
                metricType.append("FP")
        else: 
            pred_value.append(0)
            if RealLabel[i] == 0:
                metricType.append("TN")
            else:
                metricType.append("FN")

    return pred_value, metricType

#Loop from threshold array
def combinedPredictions(thresdArray, data):
    mydata = data
    DLOutputs = mydata["DLOutput"] 
    RealLabel = mydata["RealLabel"] 
    recall = []
    precision = []
    specificity = []
    F1 = []
    Accuracy = []
    balanceAccuracy = []
    length = len(thresdArray)
    for i in range (length):
        threshold  = thresdArray[i]
        predStr = "Pred" + str(threshold)
        metricStr = "Metric" +  str(threshold)
        mydata[predStr], mydata[metricStr] = predictValueWithThreshold(threshold, DLOutputs, RealLabel)
        TN = len(mydata[mydata[metricStr] == 'TN'])
        FP = len(mydata[mydata[metricStr] == 'FP'])
        TP = len(mydata[mydata[metricStr] == 'TP'])
        FN = len(mydata[mydata[metricStr] == 'FN'])
        recall.append((TP)/(TP + FN))
        precision.append((TP)/(TP + FP))
        specificity.append((TN)/(TN + FP))
        F1.append(2*((precision[i]*recall[i])/(precision[i]+recall[i])))
        Accuracy.append((TP + TN)/(TP + TN+FP + FN))
        balanceAccuracy.append((recall[i]+specificity[i])/2)
        
    return mydata, recall, precision, specificity, F1, Accuracy, balanceAccuracy 

def generateMetricTabel(thresdArray, recall, precision, specificity, F1, Accuracy, balanceAccuracy):
    print ('Predicted Class')
    resArray = [[]]
    length = len(thresdArray)
    for i in range(length):
        resArray.append([thresdArray[i], recall[i], precision[i], specificity[i], F1[i], Accuracy[i],balanceAccuracy[i]])

    myTable  = tabulate(resArray, headers=['thresdArray', 'recall', 'precision', 'specificity' , 'F1', "Accuracy",'balanceAccuracy'], tablefmt='fancy_grid')
    return myTable

