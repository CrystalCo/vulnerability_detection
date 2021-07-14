import numpy as np
import pandas as pd
import pickle
import os
from preprocess_dl_Input_version5 import *


def tranfromInput (onevectorsample, maxlen,vecdim):
    nb_samples = np.zeros((1, maxlen, vecdim))
    sequence = onevectorsample

    i = 0
    m = 0
    for vectors in sequence:#500
        n = 0
        for values in vectors:#30
            nb_samples[i][m][n] += values
            n += 1
        m += 1
         
    print(np.shape(nb_samples))
    return nb_samples


def predictLabel(mymodel, testpath, maxlen, vecdim, myoptimizer, modelname, randomSeed):
    dataset = []
    testIDs = []
    reallabels = []
    predicted_labels = []
    output_dl_labels = []
    vtype_labels = []
    for filename in os.listdir(testpath):#"./data/DLInputs/test/"
        if not filename.endswith(".DS_Store"):
            f = open(os.path.join(testpath, filename),"rb")
            dataset_file,labels_file,funcs_file,filenames_file, vtype_file, testcases_file = pickle.load(f)
            f.close()
            dataset += dataset_file
            reallabels += labels_file
            vtype_labels += vtype_file
            testIDs += testcases_file 
    if (len(dataset) == 1):#predict 1 program
        oneprogram = dataset[0]
        DL_labels = mymodel.predict(x = tranfromInput(oneprogram , maxlen, vecdim),batch_size = 1)#for 1 program
        
    else:
        myarr = process_sequences_shape(dataset, maxlen, vecdim)#predict 25 programs in files
        DL_labels = mymodel.predict(x = myarr ,batch_size = 1)
    
    for l in DL_labels:
        output_dl_labels.append(l[0])
    print(output_dl_labels[0:10])
    predicted_labels = DL_labels
    for i in range(len( DL_labels)):
        if DL_labels[i]>= 0.5:
            predicted_labels[i] = 1
        else:
            predicted_labels[i] = 0
    totalPrograms  = len(dataset)
    print("predicted array shape: ", predicted_labels.shape)#numpy array (#totalprograms, 1) like (36,1) 36 rows 1 label columns
    #transform the real labels file to the same shape of predicted for model evaluation
    new_realLabel_array = np.reshape(reallabels, (totalPrograms,1))
    new_predicted_labels = np.reshape(predicted_labels, (totalPrograms,1))
    print("new real label array shape: " , new_realLabel_array.shape)
    
    myreallabelsList = []
    mypredlabelsList = []
    myoutputlabelsList = []
    for ele in new_realLabel_array:
        myreallabelsList.append(int(ele))
    for ele in output_dl_labels:
        myoutputlabelsList.append(float(ele))
    for ele in new_predicted_labels:
        mypredlabelsList.append(int(ele))

    d = {'TestID': testcases_file, 'DLOutput':myoutputlabelsList, 'PredLabel': mypredlabelsList , 'RealLabel': myreallabelsList, 'Vtype': vtype_labels}
    df = pd.DataFrame(data=d)
    df.to_excel("OutputSummary_" + myoptimizer+ modelname + str(randomSeed) + ".xlsx")  
 
    return testcases_file , output_dl_labels, new_predicted_labels, new_realLabel_array, vtype_labels

