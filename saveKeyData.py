#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pickle

def saveKeyData(traindataSet_path):
    for filename in os.listdir(traindataSet_path):
        if filename.endswith(".DS_Store"):
            continue
        if filename.endswith(".pkl"):
            f = open(os.path.join(traindataSet_path, filename),"rb")
            print("Save metadata from filename: " + filename)
            data = pickle.load(f)
            ids = data[5]
            mytype = data[4]
            label1 = data[1]
            print(ids[:20], "\n")
            print(label1[:20], "\n")
            print(mytype[:20], "\n") 
            metadata = "caseID, realLabel, vType \n" 
            for i in range (len(ids)):
                mystr = "{},{},{}\n".format(ids[i],label1[i],mytype[i])
                metadata = metadata + mystr
            fp = open("KeyData_" + filename + ".txt", "w", encoding="utf-8")
            fp.write(metadata)
            fp.close()
            

