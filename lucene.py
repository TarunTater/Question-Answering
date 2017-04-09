import os
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
import numpy as np
import csv

# PATHS 
luceneIndexPath = '/home/tarun/PE/lucene/luceneIndexDirectory/'
corpus = '/home/tarun/PE/corpus/'
trainingFilePath = '/home/tarun/PE/Dataset/training_set.tsv'

lucene.initVM()

# ANALYZER
analyzer = StandardAnalyzer(util.Version.LUCENE_CURRENT) 

# DIRECTORY
directory = SimpleFSDirectory(File(luceneIndexPath))


# INDEX WRITER
writerConfig = IndexWriterConfig(util.Version.LUCENE_CURRENT, analyzer) 
writer = IndexWriter(directory, writerConfig)

print writer.numDocs()
# INDEXING ALL DOCUMENTS/ARTICLES IN THE CORPUS
for fileName in os.listdir(corpus):
	print fileName
	document = Document()
	article = os.path.join(corpus, fileName)
	content = open(article, 'r').read()
	document.add(Field("text", content, Field.Store.YES, Field.Index.ANALYZED))
	writer.addDocument(document)
print writer.numDocs()
writer.close()

# INDEX READER
reader = IndexReader.open(directory)
searcher = IndexSearcher(reader)

# QUERYING FOR A QUESTION
queryParser = QueryParser(util.Version.LUCENE_CURRENT, "text", analyzer)

'''
answers = ['A', 'B', 'C', 'D']
submissionFile = open("luceneModel.csv", "w")
writer = csv.writer(submissionFile, delimiter=',')
writer.writerow(['id', 'correctAnswer'])


# 10 - 0.3844
# 9 - 0.386
# 5 - 0.3742

with open(trainingFilePath) as trainData:
	reader = csv.reader(trainData, delimiter="\t")
	header=0
	accuracy = 0
	for row in reader:
		if (header == 0):
			header = 1
			continue
		else:
			question = row[1]
			answerScores = []
			for option in [row[3], row[4], row[5], row[6]]:
				# escape for handling special characters like "/"
				query = queryParser.parse(queryParser.escape(question + " " + option))
				hits = searcher.search(query, 8)
				docsScores = [hit.score for hit in hits.scoreDocs]
				answerScores.append(np.mean(docsScores))
			writer.writerow([row[0], answers[answerScores.index(np.max(answerScores))]])
			if(answers[answerScores.index(np.max(answerScores))] == row[2]):
				accuracy = accuracy+1

print accuracy*1.0/2500
'''

import pickle
with open('/home/tarun/PE/Dataset/final_test_set.pkl', 'rb') as f:
    test = pickle.load(f)

answers = ['A', 'B', 'C', 'D']
submissionFile = open("luceneModelTest.csv", "w")
writer = csv.writer(submissionFile, delimiter=',')
writer.writerow(['id', 'correctAnswer'])
for row in test:
	question = row[1]
	answerScores = []
	for option in [row[2], row[3], row[4], row[5]]:
		# escape for handling special characters like "/"
		query = queryParser.parse(queryParser.escape(question + " " + option))
		hits = searcher.search(query, 9)
		docsScores = [hit.score for hit in hits.scoreDocs]
		answerScores.append(np.mean(docsScores))
	writer.writerow([row[0], answers[answerScores.index(np.max(answerScores))]])
    