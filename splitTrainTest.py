#Split Data to Train/Test (80/20)
import pickle
import os
import numpy as np
import random
import gc
import shutil

def splitTrainTest(vType, vectorPath, vectorTrainPath, vectorTestPath, randomSeed, split = 0.8):
    
    folders = os.listdir(vectorPath)#./data/vector
    np.random.seed(randomSeed)
    np.random.shuffle(folders)
    
    folders_train = folders[:int(len(folders)*split)+1]#8090, 8091,...
    folders_test = folders[int(len(folders)*split)+1:]
    
    mode = vType
    i = 0
    train_set = [[], [], [], [], [],[]]
    train_ids = []
    #splitting method
    for folder_train in folders_train:
        if folder_train.endswith('DS_Store'):
                continue
        for filename in os.listdir(vectorPath+ folder_train + '/'):#./data/vector/8090
            if folder_train not in os.listdir(vectorTrainPath):   
                folder_path = os.path.join(vectorTrainPath, folder_train)
            f = open(vectorPath + folder_train + '/' + filename, 'rb')
            data = pickle.load(f)
            id_length = len(data[1])
            for j in range(id_length):
                train_ids.append(folder_train)
            for n in range(5):
                train_set[n] = train_set[n] + data[n]
            train_set[-1] = train_ids
        if train_set[0] == []:
            continue
    print("Samples in Train set: ",len(train_set[-1]))
    f_train = open(vectorTrainPath + mode + "_" + "train.pkl", 'wb')#api_train.pkl
    pickle.dump(train_set, f_train, protocol=pickle.HIGHEST_PROTOCOL)
    f_train.close()
    del train_set
    gc.collect()     
                    
   
    test_set = [[], [], [], [], [],[]]
    test_ids = []
    for folder_test in folders_test:
        if folder_test.endswith('DS_Store'):
            continue
        for filename in os.listdir(vectorPath + folder_test + '/'):
            if filename.endswith('DS_Store'):
                continue
            if folder_test not in os.listdir(vectorTestPath):
                folder_path = os.path.join(vectorTestPath, folder_test)
            f = open(vectorPath + folder_test + '/' + filename, 'rb')
            data = pickle.load(f)
            id_length = len(data[1])
            for j in range(id_length):
                test_ids.append(folder_test)
            for n in range(5):
                test_set[n] = test_set[n] + data[n]
                test_set[-1] = test_ids#['8090'],['8093'],['8091']
        if test_set[0] == []:
            continue
    print("Samples in Test set: " , len(test_set[-1]))  
    f_test = open(vectorTestPath + mode + "_" + "test.pkl", 'wb')
    pickle.dump(test_set, f_test, protocol=pickle.HIGHEST_PROTOCOL)
    f_test.close()
    del test_set
    gc.collect()
    print("Finished Splitting data with seed number: " , randomSeed)
    print("Train/Test Sets saved in DLVectors folder")


