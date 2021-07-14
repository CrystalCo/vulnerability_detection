import numpy as np
import pandas as pd
import os
from tabulate import tabulate
from tensorflow.keras.metrics import TruePositives, TrueNegatives,FalsePositives, FalseNegatives

def getConfusionMatrix(predicted_labels, reallabels):
    TP = TruePositives()
    TP.update_state(reallabels, predicted_labels)
    TP_count = TP.result().numpy()
    TN = TrueNegatives()
    TN.update_state(reallabels, predicted_labels)
    TN_count = TN.result().numpy()
    FP = FalsePositives()
    FP.update_state(reallabels, predicted_labels)
    FP_count = FP.result().numpy()
    FN = FalseNegatives()
    FN.update_state(reallabels, predicted_labels)
    FN_count = FN.result().numpy()

    totalsamples = (TP_count + TN_count + FP_count + FN_count)
    accuracyrate = (TP_count + TN_count) /totalsamples 
    sensitivity = TP_count/(TP_count + FN_count)
    specificity = TN_count/(TN_count + FP_count)
    precision = TP_count/(TP_count + FP_count)
    negPrediction = TN_count/(TN_count + FN_count)
    print ('Predicted Class')
    print('Total Samples', totalsamples  )
    print(tabulate([['Positive', TP_count, FN_count], ['Negative', FP_count, TN_count ]], headers=['Type', 'Positive', 'Negative'], tablefmt='orgtbl'))
    print()
    print ('Predicted Class')
    print(tabulate([['Positive', TP_count, FN_count, sensitivity, 'Sensitivity' ], 
                ['Negative', FP_count, TN_count,specificity, 'specificity' ],
                ['', precision, negPrediction, accuracyrate, 'Accuracy' ],
                ['', 'Precision', 'NegPrediction', '', '' ]
                
               ], headers=['', 'Positive', 'Negative', 'Rate' , ''], tablefmt='fancy_grid'))

