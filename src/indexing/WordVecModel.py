#TODO: use self.text (which is what is only needed) for training to build the model, have buildmodel, saveModel, loadModel

import re,sys
import pysubs2
import math
import string
# -*- coding: utf-8 -*- and from __future__ import division, unicode_literals
from textblob import TextBlob as tb
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import array
from gensim.models.doc2vec import LabeledSentence
from PorterStemmer import PorterStemmer
from nltk.corpus import stopwords
from gensim import models
from random import shuffle


import gensim, logging
import os
import random
from collections import defaultdict
from sklearn.metrics.pairwise import linear_kernel
import csv

#self.buildWordVec(text) :Training using self.text

class WordVec:
    def __init__(self, text = None):
        self.wordvecModel = None
        if text != None:
            self.buildWordVec(text)
        #self.vecToTime = None

    #prints out the most similar docs given a list of similar docs
    #each doc is a tuple: (SRT_(.*?.srt)_(.*?)_(.*?), similarityScore)
    def buildPreTrainedVec(self):
        preTrainedModel = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
        preTrainedModel.save('srtModel')

    #builds word vecotr using given text
    def buildWordVec(self, text):
        #print self.text
        #preTrainedModel = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
        #preTrainedModel.build_vocab(self.text, update=True)

        self.wordvecModel = gensim.models.Word2Vec(text, min_count=1)
        #model.reset_from(preTrainedModel)
        #preTrainedModel.train(self.text)
        #preTrainedModel.save('srtModel') #model for word2vec from syllabus

    #saves the wordvecModel
    def saveWordVec(self, fname = 'wordVecModel_SRT'):
        self.wordvecModel.save(fname)
        print("Done saving wordVec")

    #loads a given word2vec Model
    def loadWordVec(self, modelName = 'wordVecModel_SRT'):
        self.wordvecModel = gensim.models.Word2Vec.load(modelName)
        #connects pretrained word to timeslot and correpsonds this information for wordvector to timeslot
        #for word in self.wordToTime:
        #    self.vecToTime[tuple(self.wordvecModel[word])] = self.wordToTime[word]

    #prints similar words using randomized queries
    def getRandomQuery(self):
        word = random.choice(list(self.uniqueVocab))
        if self.wordvecModel is None:
            self.wordvecModel = gensim.models.Word2Vec.load('wordVecModel')
        print 'Time slots'
        print self.wordToTime[word]
        print self.vecToTime[tuple(self.wordvecModel[word])]
        similarList = self.wordvecModel.most_similar(positive=[word], topn=10)
        print 'Randomly chosen word and similar lists'
        print word
        print similarList
        #>>> model.score(["The fox jumped over a lazy dog".split()]) score probability of a query

    #test for each query
    def testQuery(self, query):
        phrase = query.split()
        phraseList = [w.strip() for w in phrase]
        similarList = self.wordvecModel.most_similar(positive = phraseList, negative = [], topn=5)
        print query
        print similarList

    #sample testQueries
    def testQueries(self):
        # add support for phrase queries
        #add support for unseen words
        #use tfidf to rank important words usseful for ranking and selecting important word also for querying
        print 'Word2vec test'
        wordList = ['recursion', 'variable', 'for loop' , 'classes', 'network', 'condition', 'function', 'computer science', 'arithmetic']
        #print(model.most_similar(positive=['computer', 'science'], negative=[], topn=5)) #concatenates and prints out top 5 similar words [useful for study guides]

        for word in wordList:
            testQuery(word)

def main():
    parser = word2vec()
    parser.saveModel()

if __name__ == "__main__":
    main()
