import gensim
import os
import nltk
from nltk.corpus import stopwords
import csv
import numpy as np

trainingFilePath = '/home/tarun/PE/Dataset/training_set.tsv'

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

#sentences = MySentences('/home/tarun/PE/corpus/') # a memory-friendly iterator
#model = gensim.models.Word2Vec(sentences, workers=4) # min_count=5, size=100,
#model.save('word2vecModel')

model = gensim.models.Word2Vec.load('word2vecModel')

failed = 0

def getSimilarity(sentence1, sentence2):
	global failed
	tokens1 = nltk.word_tokenize(sentence1)
	tokens2 = nltk.word_tokenize(sentence2)
	try:
		tokens1 = filter(lambda x: x in model.vocab, tokens1)		
		tokens2 = filter(lambda x: x in model.vocab, tokens2)
		nonStopWords1 = [word for word in tokens1 if word not in stopwords.words('english')]
		nonStopWords2 = [word for word in tokens2 if word not in stopwords.words('english')]
		wordSimScores = []
		for word1 in nonStopWords1:
			for word2 in nonStopWords2:
				wordSimScores.append(model.similarity(word1, word2))
		if (len(wordSimScores) == 0):
			return 0
		return np.mean(wordSimScores)
	except:
		failed+=1
	return 0

'''
# TRAINING ACCURACY
submissionFile = open("word2vecModel.csv", "w")
writer = csv.writer(submissionFile, delimiter=',')
writer.writerow(['id', 'correctAnswer'])	
answers = ['A', 'B', 'C', 'D']
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
				answerScores.append(getSimilarity(question, option))
			writer.writerow([row[0], answers[answerScores.index(np.max(answerScores))]])
			if(answers[answerScores.index(np.max(answerScores))] == row[2]):
				accuracy = accuracy+1
				
print accuracy*1.0/2500

'''
# TESTING
import pickle
with open('/home/tarun/PE/Dataset/final_test_set.pkl', 'rb') as f:
    test = pickle.load(f)

answers = ['A', 'B', 'C', 'D']
submissionFile = open("word2vecModelTest.csv", "w")
writer = csv.writer(submissionFile, delimiter=',')
writer.writerow(['id', 'correctAnswer'])
checkRows = []
for row in test:
	question = row[1]
	answerScores = []
	for option in [row[2], row[3], row[4], row[5]]:
		similarity = getSimilarity(question, option)
		answerScores.append(similarity)
		if similarity == 0:
			checkRows.append(test.index(row))
	writer.writerow([row[0], answers[answerScores.index(np.max(answerScores))]])

print checkRows
