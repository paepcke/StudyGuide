# things to do with wordvectors: https://radimrehurek.com/gensim/models/keyedvectors.html
# model['computer']  # raw numpy vector of a word, with trained word2vec
#includes genism for phrase2vec
#frequency of words for the whole file --> could be in the list for stop Words
#import youtube api
#porter stemming: for the whole corpus before indexing https://pythonprogramming.net/stemming-nltk-tutorial/ ***
#stop-wrds ****
#youtube api ******
#forum posts for querying data
#taking queries from random subtititles in srt file as queries ***
#query expansion: using word2vec to find out most similar words from query to retrieve snippet *****
#querying for the timestamp the word occurs with each second having probability of each word in |V| occurring
#get related courses from syllabi through querying from one specific course ****
#at each second having probability of all words in |V|: column vectorizer
#can get most important words i.e. words having highest probabilities, in a certain time stretch
#can be used for table of contents or important words student needs to know
#fastText (now also integrated into gensim): break the unknown word into smaller character n-grams. Assemble the word vector from vectors of these ngrams.
#deal with unseen words


#phrase2vec, doc2vec
#get closest word, if not in word, then go to next closest
#stopwords: english online
#tfidf to get the highest ranked words, and also get stopwords through dissimilar videos srt
#tfidf query (noun phrases,etc)
#training
#ranking of the video snippets indexed by the user through survey
#tfidf of 1 minute time slots

#remove words with count of 1
#find snippet of when this explanation started

import re,sys
import pysubs2
from numpy import array
from PorterStemmer import PorterStemmer
from nltk.corpus import stopwords
from gensim import models


import gensim, logging
import os
import random

class parseSrt:
    def __init__(self):
        self.listFiles = ['closed_caption.srt']
        self.wordToTime = {}
        self.vecToTime = {}
        self.text = []
        self.new_model = None
        self.alphanum = re.compile('[^a-zA-Z0-9]')
        self.p = PorterStemmer()
        self.uniqueVocab = set()
        self.stop_words = set(stopwords.words('english'))
        self.path = './srtFiles/'


    def getRandomQuery(self):
        word = random.choice(list(self.uniqueVocab))
        self.new_model = gensim.models.Word2Vec.load('srtModelSmall')
        print 'Time slots'
        print self.wordToTime[word]
        print self.vecToTime[tuple(self.new_model[word])]
        similarList = self.new_model.most_similar(positive=[word], topn=10)
        print 'Randomly chosen word and similar lists'
        print word
        print similarList
        print '\nSample tests:'
        self.testQueries()
        #>>> model.score(["The fox jumped over a lazy dog".split()]) score probability of a query

    def testQueries(self):
        # add support for phrase queries
        #add support for unseen words
        #use tfidf to rank important words usseful for ranking and selecting important word also for querying
        wordList = ['loop', 'variable', 'for loop' , 'classes', 'if', 'condition', 'function', 'computer science', 'arithmetic']
        #print(model.most_similar(positive=['computer', 'science'], negative=[], topn=5)) #concatenates and prints out top 5 similar words [useful for study guides]

        for word in wordList:
            phrase = word.split()
            phraseList = [w.strip() for w in phrase]

            similarList = self.new_model.most_similar(positive = phraseList, negative = [], topn=5)
            print word
            print similarList

    def openFile(self):
        # TODO looking at data: deal with stop words, deal with commas

        for fileName in os.listdir(self.path):
             fileName = self.path + fileName
             #print fileName
        #for fileName in self.listFiles:
             subs = pysubs2.load(fileName)
             for i in xrange(len(subs)):
                 startTimeStamp = subs[i].start
                 endTimeStamp = subs[i].end
                 sentence = []
                 line = subs[i].text
                 line = line.replace('\\N',' ')
                 line = line.replace('\\n',' ')
                 line = line.split()
                  # remove non alphanumeric characters
                 line = [self.alphanum.sub('', xx) for xx in line]
                  # remove any words that are now empty
                 line = [xx.strip() for xx in line if xx != '']
                  # stem words
               # line = [self.p.stem(xx) for xx in line if xx not in self.stop_words]
                 line = [xx for xx in line if xx not in self.stop_words]
                 for word in line:
                     #print word
                     word = word.lower()
                     self.uniqueVocab.add(word)
                     timeList = []
                     if word in self.wordToTime:
                         timeList = self.wordToTime.get(word)
                     timeList.append((startTimeStamp, endTimeStamp, fileName))
                     self.wordToTime[word] = timeList
                     sentence.append(word)
                 self.text.append(sentence)

    def buildPreTrainedVec(self):
        preTrainedModel = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
        preTrainedModel.save('srtModel')

    def buildWordVec(self):
        #print self.text
        #preTrainedModel = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
        #preTrainedModel.build_vocab(self.text, update=True)

        model = gensim.models.Word2Vec(self.text, min_count=1)
        model.save('srtModelSmall')
        #model.reset_from(preTrainedModel)
        #preTrainedModel.train(self.text)
        #preTrainedModel.save('srtModel') #model for word2vec from syllabus
        print("Done")

    def loadWordVec(self):
        new_model = gensim.models.Word2Vec.load('srtModelSmall')
        for word in self.wordToTime:
            self.vecToTime[tuple(new_model[word])] = self.wordToTime[word]
        #print self.vecToTime

def main():
    parser = parseSrt()
    parser.openFile()
    parser.buildWordVec()
    parser.loadWordVec()
    parser.getRandomQuery()

if __name__ == "__main__":
    main()
