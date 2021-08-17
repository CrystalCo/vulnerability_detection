from tensorflow.keras.metrics import TruePositives, TrueNegatives,FalsePositives, FalseNegatives
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad, Adam, Adadelta
from keras.models import Sequential, load_model
from keras.layers.core import Masking, Dense, Dropout, Activation
from keras.layers.recurrent import LSTM,GRU
from keras.layers.wrappers import Bidirectional
from collections import Counter
from keras import optimizers 
from preprocess_dl_Input_version5 import *
import numpy as np
import pickle
import random
import time
import math
import os

def buildBGRU(maxlen, vector_dim, layers, dropout, myoptimizer): 
    print('Build model...')
    model = Sequential()
    
    model.add(Masking(mask_value=0.0, input_shape=(maxlen, vector_dim)))
    
    for i in range(1, layers):
        model.add(Bidirectional(GRU(units=256, activation='tanh', recurrent_activation='hard_sigmoid', return_sequences=True)))
        model.add(Dropout(dropout))
        
    model.add(Bidirectional(GRU(units=256, activation='tanh', recurrent_activation='hard_sigmoid')))
    model.add(Dropout(dropout))
    
    model.add(Dense(1, activation='sigmoid'))
    
    TP_count = TruePositives()
    TN_count = TrueNegatives() 
    FP_count = FalsePositives() 
    FN_count = FalseNegatives()
          
    #model.compile(loss='binary_crossentropy', optimizer='sgd', metrics= ['accuracy', TP_count, TN_count, FP_count, FN_count])
    #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    
    model.compile(loss='binary_crossentropy', optimizer = myoptimizer, metrics=['accuracy'])
    model.summary()
 
    return model

from tensorflow.keras.metrics import TruePositives, TrueNegatives,FalsePositives, FalseNegatives
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad, Adam, Adadelta
from keras.models import Sequential, load_model
from keras.layers.core import Masking, Dense, Dropout, Activation
from keras.layers.recurrent import LSTM,GRU
from preprocess_dl_Input_version5 import *
from keras.layers.wrappers import Bidirectional
from collections import Counter
from keras import optimizers 
import numpy as np
import pickle
import random
import time
import math
import os

def buildBLSTM(maxlen, vector_dim, layers, dropout, myoptimizer): 
    print('Build model...')
    model = Sequential()
    
    model.add(Masking(mask_value=0.0, input_shape=(maxlen, vector_dim)))
    
    for i in range(1, layers):
        model.add(Bidirectional(LSTM(units = 256, activation='tanh', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=2, return_sequences=True, return_state=False, go_backwards=False, stateful=False, unroll=False)))
        
    model.add(Bidirectional(GRU(units=256, activation='tanh', recurrent_activation='hard_sigmoid')))
    model.add(Dropout(dropout))
    
    model.add(Dense(1, activation='sigmoid'))
    
    TP_count = TruePositives()
    TN_count = TrueNegatives() 
    FP_count = FalsePositives() 
    FN_count = FalseNegatives()
          
    #model.compile(loss='binary_crossentropy', optimizer='sgd', metrics= ['accuracy', TP_count, TN_count, FP_count, FN_count])
    #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    
    model.compile(loss='binary_crossentropy', optimizer = myoptimizer, metrics=['accuracy'])
    model.summary()
 
    return model




def fitModel(myKerasModel,  weightpath, traindataSet_path, batch_size, maxlen, vector_dim, RANDOMSEED):
    
    print("Fit model with Trainning set...")
    dataset = []
    labels = []
    testcases = []
    for filename in os.listdir(traindataSet_path):
        if not filename.endswith(".DS_Store"):
            print(filename)
            f = open(os.path.join(traindataSet_path, filename),"rb")
            dataset_file,labels_file,funcs_file,filenames_file,vtype_file, testcases_file = pickle.load(f)
            f.close()
            dataset += dataset_file
            labels += labels_file
            testcases_file
    print(len(dataset), len(labels))

    bin_labels = []
    for label in labels:
        bin_labels.append(multi_labels_to_two(label))
    labels = bin_labels
    
    np.random.seed(RANDOMSEED)
    np.random.shuffle(dataset)
    np.random.seed(RANDOMSEED)
    np.random.shuffle(labels)
   
    train_generator = generator_of_data(dataset, labels, batch_size, maxlen, vector_dim)   
    all_train_samples = len(dataset)
    steps_epoch = int(all_train_samples / batch_size)
    print("start")
    t1 = time.time()
    myKerasModel.fit_generator(train_generator, steps_per_epoch=steps_epoch, epochs=10)
    t2 = time.time()
    train_time = t2 - t1
    myKerasModel.save_weights(weightpath)
    return myKerasModel

