import pickle
import os
import numpy as np
import random
import shutil
import gc

def appendCaseIDLabel0 (vectorTrainPath):
    for file in os.listdir(vectorTrainPath):
        if file.endswith('.pkl'):
            print(vectorTrainPath + file)
            with open(vectorTrainPath + file, 'rb') as f:
                data = pickle.load(f)
                print("Elements in each vector .pkl file: ", len(data))
                label_array = data[1]
                caseID_array = data[5]
                caseID_zero = []
                caseID_one = []
                label_one = []
                count_1 = 0
                count_0 = 0
                totalsamples = len(label_array)
                for i in range (totalsamples):
                    if  label_array[i] ==1:
                        count_1 = count_1 + 1
                        caseID_one.append(caseID_array[i])
                    else:
                        count_0 = count_0 + 1
                        caseID_zero.append(caseID_array[i])#contain case id with class label 1
          
    return  caseID_one, caseID_zero, count_1

def downsampling (caseID_one, caseID_zero, downsampleNum, seed,  vectorPath, trainpath):     	    
    print("Downsampling with seed number: " , seed)
    np.random.seed(seed)
    randomCase = caseID_zero
    np.random.shuffle(randomCase)
    reducedCase_0 = randomCase[0:downsampleNum]
    combinedCaseID = caseID_one + reducedCase_0
    np.random.seed(seed)
    randomCombinedCase = combinedCaseID
    np.random.shuffle(randomCombinedCase)
    
    train_set = [[], [], [], [], [], []]
    
    for foldername in os.listdir(vectorPath):  #./data/DLvectors/train/
        if foldername.endswith('.pkl') or foldername.endswith('DS_Store'):#loop folder
            continue
        else:
            for caseID in randomCombinedCase:
                if (caseID == foldername):
                    for pkl in os.listdir(vectorPath + foldername):
                        if pkl.endswith('.pkl'):
                            f = open(vectorPath + foldername + '/'  + pkl , 'rb') #'./data/vector/8508/8508.pkl'
                            data = pickle.load(f)
                            train_set[0].append(data[0][0])
                            train_set[1].append(data[1][0])
                            train_set[2].append(data[2][0])
                            train_set[3].append(data[3][0])
                            train_set[4].append(data[4][0])
                            train_set[5].append(foldername)
   
    f_train = open(trainpath + "balancedClassTrain.pkl", 'wb')
    pickle.dump(train_set, f_train, protocol=pickle.HIGHEST_PROTOCOL)
    f_train.close()
    del train_set
    gc.collect() 
    print("Done! balancedClassTrain.pkl saved in ./data/DLvectors/train/ ")

def isClassBalanced(traindatapath):
    count_1 = 0
    count_0 = 0
    for filename in os.listdir(traindatapath):  #./data/DLvectors/train/
        if filename.endswith('DS_Store'):#loop folder
            continue
        if ("balanced" in filename) and (filename.endswith('.pkl')):#loop for each train.pkl and extrak class.pkl
            print(filename)
            f = open(traindatapath + filename, 'rb') #'./data/DLvectors/train/8508/api8508.pkl'
            data = pickle.load(f)
            label_array = data[1]
            totalsamples = len(label_array)
            for i in range (totalsamples):
                if label_array[i] == 1:
                    count_1 = count_1 + 1
                else:
                    count_0 = count_0 + 1
            print ("total label 0: " ,count_0)
            print ("total label 1: " ,count_1)
            print ("total sample: ", count_0 + count_1 )
    if (count_0 == count_1):
        return True
    else:
        return False
