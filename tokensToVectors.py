#Iterate each .pkl file and create DirofCorpus(object) from tokens to fit the W2V Function
from gensim.models.word2vec import Word2Vec
import pickle
import os
import gc

class DirofCorpus(object):
    def __init__(self, dirname):
        self.dirname = dirname
    
    def __iter__(self):
        d = self.dirname
        for fn in os.listdir(d): 
            if not fn.endswith('DS_Store'):
                fnn = fn
                for filename in os.listdir(os.path.join(d, fnn)):
                    if not filename.endswith('DS_Store'):
                        pklname = filename
                        with open(os.path.join(d, fnn, pklname), 'rb') as f:
                            samples = pickle.load(f)[0]
                            for sample in samples:
                                #print (sample)
                                yield sample
                            del samples
                            gc.collect()

def createW2VModel(w2vmodelPath, tokenPath, mysize):
    print("Fitting W2V model from corpus...")
    mymodel = Word2Vec(sentences=DirofCorpus(tokenPath) , size=mysize, alpha=0.01, window=5, min_count=0, max_vocab_size=None, sample=0.001, seed=1, workers=1, min_alpha=0.0001, sg=1, hs=0, negative=10, iter=1)
    mymodel.save(w2vmodelPath)
    print("Model created and saved in: " + w2vmodelPath)
    words = sorted(mymodel.wv.vocab.keys())
    print("Number of words in model:", len(words))
    fp = open("wordsW2Vmodel.txt", "w", encoding="utf-8")
    for word in words:
        fp.write(word + '\n')
    fp.close()
    return mymodel

def fitW2VModel(w2vmodelPath, tokenPath, vectorPath):
    mymodel = Word2Vec.load(w2vmodelPath)
    for corpusfiles in os.listdir(tokenPath):
        #print(corpusfiles)
        if not corpusfiles.endswith('DS_Store'):
            cfs = corpusfiles
            if cfs not in os.listdir(vectorPath): 
                folder_path = os.path.join(vectorPath, cfs)
                os.mkdir(folder_path)
            for corpusfile in os.listdir(tokenPath + cfs):
                if not corpusfile.endswith('DS_Store'):
                    cf = corpusfile
                    corpus_path = os.path.join(tokenPath, corpusfiles, cf)
                    f_corpus = open(corpus_path, 'rb')
                    data = pickle.load(f_corpus)
                    f_corpus.close()
                    samples = data[0]
                    data[0] = [[mymodel[word] for word in sample] for sample in samples]
                    vector_path = os.path.join(vectorPath, corpusfiles, corpusfile)
                    f_vector = open(vector_path, 'wb')
                    pickle.dump(data, f_vector, protocol=pickle.HIGHEST_PROTOCOL)
                    f_vector.close()
    print("W2V Completed: The vector file is in vector folder")
