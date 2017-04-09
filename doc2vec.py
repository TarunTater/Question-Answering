import nltk
from nltk.corpus import stopwords

import numpy as np
import os
import os.path
import string
import re
import operator
import csv
import logging
import time
import random
import pickle
import glob

import gensim
import gensim.models as g

import lucene

from java.io import File 
from org.apache.lucene.analysis.standard import StandardAnalyzer 
from org.apache.lucene.document import Document, Field 
from org.apache.lucene.index import IndexWriter, IndexWriterConfig 
from org.apache.lucene.store import SimpleFSDirectory 
from org.apache.lucene.util import Version
from org.apache.lucene.search import IndexSearcher 
from org.apache.lucene.index import IndexReader 
from org.apache.lucene.queryparser.classic import QueryParser 

from org.apache.lucene import document, store, util

from keras.models import Sequential, Model, Graph
from keras.layers import Dense, Activation, Input, Embedding, LSTM, merge, Dropout
from keras.regularizers import l2, activity_l2
from keras.layers.core import Dense, Flatten, Merge
from keras.utils.visualize_util import plot

from scipy import spatial

def tokenizeSentence(fileText):
    fileText = fileText.replace('\n', ' ') # each file's text - a paragraph like structure for us
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        fileText = fileText.replace(char, '')
    fileText = fileText.replace("displaystyle", '')
    fileText = re.sub("""displaystyle""", "", fileText, re.I|re.S)
    fileText = re.sub("""[^0-9A-Za-z]""", ' ', fileText)

    fileText = re.sub("""\s+""", " ", fileText)
    fileText = re.sub("""\t+""", " ", fileText)
    tokens1 = nltk.word_tokenize(fileText)
    return tokens1
    
###################### Loading the trained Doc2vec Model

#input corpus
#train_corpus = "/home/tarun/PE/allNewCorpus100Text.txt"

#output model
modelPath = "/home/tarun/PE/doc2vec/model3_100_newCorpus100_1min_6window_100trainEpoch.bin"

model = g.Doc2Vec.load(modelPath)

###################### end 

###################### retrieving using Lucene

# PATHS 
luceneIndexPath = '/home/tarun/PE/lucene/luceneIndexDirectoryNewCorpus100/'
corpus = '/home/tarun/PE/newCorpus100/'
#trainingFilePath = '/home/madhumathi/sem9/PE/models/Dataset/training_set.tsv'
  
lucene.initVM()

# ANALYZER
analyzer = StandardAnalyzer(util.Version.LUCENE_CURRENT) 

# DIRECTORY
directory = SimpleFSDirectory(File(luceneIndexPath))


#dont forget to remove the luceneIndexDirectory file everytime you run this code.
'''
# INDEX WRITER
code removed
'''
# INDEX READER
reader = IndexReader.open(directory)
searcher = IndexSearcher(reader)

# QUERYING FOR A QUESTION
queryParser = QueryParser(util.Version.LUCENE_CURRENT, "text", analyzer)

###################### end

###################### creating vectors (infering) for testing

#inference hyper-parameters
tic = time.time()
start_alpha=0.01
infer_epoch=1000
testingFilePath = '/home/tarun/PE/testFiles/testNum18000.csv'

#answers = ['A','B','C','D']

with open(testingFilePath) as f:
    reader = csv.reader(f,delimiter = ",")
    data = list(reader)
    numQuestions = len(data) - 1

with open(testingFilePath) as testData:
    reader = csv.reader(testData, delimiter=",")
    header = 0
    #storeInputVecInFile = np.zeros([numQuestions,700]) #question, docs, options
    #storeOutputVecInFile = np.zeros([numQuestions,4])
    test_inputDocVecs = []
    test_inputQuesVecs = []
    test_inputOptionVecs = []
    #outputVecs = []
    inputNum = 0
    for row in reader:
        question = row[1]
        
        #print question, "\n\n"
        query = queryParser.parse(queryParser.escape(question))
        question = tokenizeSentence(question)
        questionVec = model.infer_vector(question, alpha=start_alpha, steps=infer_epoch)
        numPages = 1
        hits = searcher.search(query, numPages)
        docText = ""
        output = []
        for hit in hits.scoreDocs:
            doc_id = hit.doc
            #print doc_id, hit.toString()
            docT = searcher.doc(hit.doc)
            docText = docT.get("text").encode("utf-8")
        
        #print docText, "\n\n"
        listOfWords = docText.split(" ")
        numWords = len(listOfWords)
        vector_size = 100
        num = 0

        docVec = np.zeros([numWords,vector_size])
        for i in range(1, numWords+1):    
            wordsTillNow = " ".join(listOfWords[0:i])
            docV = model.infer_vector(tokenizeSentence(wordsTillNow), alpha=start_alpha, steps=infer_epoch)

            #docV = model.infer_vector(tokenizeSentence(wordsTillNow))
            docVec[num] = docV
            num = num + 1
        test_inputDocVecs.append(docVec)
        test_inputQuesVecs.append(questionVec)
        optionList = [row[2], row[3], row[4], row[5]]
        #inputVec = np.concatenate([docVec, questionVec])
        output = []
        options = []
        for option in optionList:
            optionVec = model.infer_vector(tokenizeSentence(option), alpha=start_alpha, steps=infer_epoch)
            options.append(optionVec)
            #if(answers[optionList.index(option)] == row[2]):
            #    output.append(1)
            #else:
            #    output.append(0)
                
            #inputVec = merge([docVec, questionVec, optionVec], mode='concat')
            #inputVec = np.concatenate([inputVec, optionVec])
        test_inputOptionVecs.append(options)
       
        inputNum = inputNum + 1
        if(inputNum % 50 == 0):
            print inputNum, (time.time() - tic)
# Save the input vectors and output into a file
#model3_100_CBSESmallFiles_1min_6window_100trainEpoch
np.save('test18000_inputDocVecs_model3_100_newCorpus100_1min_6window_1000inferEpoch.npy', test_inputDocVecs)
np.save('test18000_inputQuesVecs_model3_100_newCorpus100_1min_6window_1000inferEpoch.npy', test_inputQuesVecs)
np.save('test18000_inputOptionVecs_model3_100_newCorpus100_1min_6window_1000inferEpoch.npy', test_inputOptionVecs)
#np.save('outputVecs_model3_100_CBSESmallFiles_1min_6window_1000inferEpoch.npy', outputVecs)

toc = time.time()
print toc - tic

###################### end


