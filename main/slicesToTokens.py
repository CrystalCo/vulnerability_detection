import os
import pickle
from mapping import *

def tokenizeSlices(slicepath, corpuspath, totalSamples):
    
    mycase_ID = []
    for filename in os.listdir(slicepath):
        if(filename.endswith(".txt") is False):
            continue
        print("Slice Files To be Processed: ", filename)
        if "API" in filename:
            myType = ['API']
        elif "Array" in filename:
            myType = ['ARR']
        elif "Pointer" in filename:
            myType = ['PTR']
        elif "Arithmetic" in filename:
            myType = ['AE']
        else:
            myType = 'Others'
        filepath = os.path.join(slicepath, filename)
        f1 = open(filepath, 'r')
        slicelists = f1.read().split("------------------------------")#aa. split each slide (each program in .txt file)
        f1.close()

        if slicelists[0] == '':
            del slicelists[0]
        if slicelists[-1] == '' or slicelists[-1] == '\n' or slicelists[-1] == '\r\n':
            del slicelists[-1]

        lastprogram_id = 0
        program_id = 0
        index = -1
        file_name = 0
        slicefile_corpus = []
        slicefile_labels = []
        slicefile_func = []
        slicefile_filenames = []
        focuspointer = None 
        count = 0
        for slicelist in slicelists:
            if count == totalSamples:
                continue
            count = count+1
            slice_corpus = []
            index = index + 1
            sentences = slicelist.split('\n')
            if sentences[0] == '\r' or sentences[0] == '':
                del sentences[0]
            if sentences == []:
                continue
            if sentences[-1] == '':
                del sentences[-1]
            if sentences[-1] == '\r':
                del sentences[-1]
            testcase_id = sentences[0].split(' ')[0] 
            if testcase_id in os.listdir(corpuspath):
                testcase_id = int(testcase_id) + 1000000
            else:
                testcase_id = int(testcase_id)
            mycase_ID.append(testcase_id)
            label = int(sentences[-1])
            program_id = testcase_id 
            lastprogram_id = testcase_id
            focuspointer = sentences[0].split(" ")[-2:]
            sliceid = index
            file_name = sentences[0]
            folder_path = os.path.join(corpuspath, str(lastprogram_id))
            filenameCorpus = str(testcase_id)
            savefilename = folder_path + '/' + filenameCorpus + '.pkl'
            if lastprogram_id not in os.listdir(corpuspath):
                os.mkdir(folder_path)
            slicefile_corpus = []
            slicefile_labels = []
            slicefile_filenames = []
            slicefile_func = []
            sentences = sentences[1:]
            for sentence in sentences:

                if sentence.split(" ")[-1] == focuspointer[1] and flag_focus == 0:
                    flag_focus = 1

                sentence = ' '.join(sentence.split(" ")[:-1])

                start = str.find(sentence,r'printf("')
                if start != -1:
                    start = str.find(sentence,r'");')
                    sentence = sentence[:start+2]
                
                fm = str.find(sentence,'/*')
                if fm != -1:
                    sentence = sentence[:fm]
                else:
                    fm = str.find(sentence,'//')
                    if fm != -1:
                        sentence = sentence[:fm]
                sentence = sentence.strip()
                list_tokens = create_tokens(sentence)
                slice_corpus.append(list_tokens)

            slicefile_labels.append(label)
            slicefile_filenames.append(file_name)
            
            slice_corpus = mapping(slice_corpus)
            slice_func = slice_corpus
            slice_func = list(set(slice_func))
                   
            if slice_func == []:
                slice_func = ['main']
            sample_corpus = []
            for sentence in slice_corpus:
                list_tokens = create_tokens(sentence)
                sample_corpus = sample_corpus + list_tokens
            slicefile_corpus.append(sample_corpus)
            slicefile_func.append(slice_func)
            f1 = open(savefilename, 'wb')               
            pickle.dump([slicefile_corpus,slicefile_labels,slicefile_func,slicefile_filenames, myType], f1)
            f1.close()
        print ("Total Corpus Files: ", count)
        print ("Last Program ID: ", lastprogram_id)
    return mycase_ID

