from __future__ import absolute_import
from __future__ import print_function
import pickle
import numpy as np
import random
import time
import math
import os
from collections import Counter

def meanLen(train_Path):
    print("Loading data...")
    totalsamples = 0
    totalvectorlen = 0
    meanLen = 0
    for filename in os.listdir(train_Path):
        if not (filename.endswith(".pkl")):
            continue
        if not ("balanced" in filename) :
            continue
        dataSetpath = train_Path + filename
        print (dataSetpath)
        f1 = open(dataSetpath, 'rb')
        X, ids, funcs, filenames, vtype, test_cases = pickle.load(f1)
        f1.close()
        length = len(X) #total samples for 1 file: 516
        for i in range(length): 
                #print(len(X[i]))#vector len of each program
            totalvectorlen = totalvectorlen + len(X[i])
    totalsamples = length 
    meanLen = int(totalvectorlen/totalsamples)
    print("Mean Vector Length" ,  meanLen)
            
    return meanLen


def load_data_binary(dataSetpath, maxlen, vector_dim):   
    f1 = open(dataSetpath, 'rb')
    print(dataSetpath)
    data = pickle.load(f1)
    X, ids, funcs, filenames, vtypes, test_cases = data[0], data[1], data[2], data[3], data[4], data[5]
    f1.close()
    cut_count = 0
    fill_0_count = 0
    no_change_count = 0
    fill_0 = [0]*vector_dim
    totallen = 0
    threshold = maxlen * vector_dim
    print ("threshold: " ,threshold)
    if (maxlen != 0):
        new_X = []
        for x, i, func, file_name, vtype, test_case in zip(X, ids, funcs, filenames, vtypes, test_cases):  
             #len(x) is how many symbols in 1 program. ex. 79 
            if len(x) <  maxlen:
                x = x + [fill_0] * (maxlen - len(x))
                new_X.append(x)
                fill_0_count += 1

            elif len(x) == maxlen:
                new_X.append(x)
                no_change_count += 1
                    
            else:
                startpoint = int(threshold - round(maxlen / 2.0))
                endpoint =  int(startpoint + maxlen)
                if startpoint < 0:
                    startpoint = 0
                    endpoint = maxlen
                if endpoint >= len(x):
                    startpoint = -maxlen
                    endpoint = None
                new_X.append(x[startpoint:endpoint])
                cut_count += 1
            totallen = totallen + len(x)
    X = new_X
    print ("New Vector Length: ", len(X[0]))
    return X, ids, funcs, filenames, vtypes, test_cases


def tranformVectorLen(raw_traindataSet_path, raw_testdataSet_path, traindataSet_path, testdataSet_path, maxLen, vector_dim, vType):
    
    print("Loading data...")
    print("Train set")
    for filename in os.listdir(raw_traindataSet_path):
        if not (filename.endswith(".pkl")):
            continue
        if not ("balanced" in filename):
            continue
        X_train, train_labels, funcs, filenames, vtypes, testcases = load_data_binary(raw_traindataSet_path + filename, maxLen, vector_dim)
        f_train = open(traindataSet_path + "DL_Final_" + filename , 'wb')
        pickle.dump([X_train, train_labels, funcs, filenames, vtypes, testcases], f_train)
        f_train.close()
    
    print("\nTest set")
    for filename in os.listdir(raw_testdataSet_path):
        if not (vType in filename):
            continue
        if not (filename.endswith(".pkl")):
            continue
        X_test, test_labels, funcs, filenames, vtypes, testcases = load_data_binary(raw_testdataSet_path + filename, maxLen, vector_dim)
        f_test = open(testdataSet_path + "DL_Final_" + filename , 'wb')
        pickle.dump([X_test, test_labels, funcs, filenames, vtypes, testcases], f_test)
        f_test.close()

