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

#self.trainDoc2Vec(self.labeledSentences) :Training
#self.getMostSimilarDocs(query) : Get list of most similar docs given a string query
#self.printSimilarDocs(similarDocs, docToTime): prints out the most similar douments given similar docList, docToTime(document to time interval)

class DocVec:
    def __init__(self, tagged_sentences = None, docToTime = None):
        self.doc2vecModel = None
        self.tagged_sentences = None
        #print tagged_sentences
        #print docToTime
        if tagged_sentences != None and docToTime != None:
            self.trainDoc2Vec(tagged_sentences)
            self.docToTime = docToTime
    #prints out the most similar docs given a list of similar docs
    #each doc is a tuple: (SRT_(.*?.srt)_(.*?)_(.*?), similarityScore)

    def testQuery(self, query):
        simDocList = self.getMostSimilarDocs(query)
        self.printSimilarDocs(simDocList)

    def printSimilarDocs(self, simDocList):
        if self.docToTime is None:
            print 'Error! Document-Time indexing has not been initialized.'
        else:
            for doc in simDocList:
                fileName = ""
                startTimeStamp = ""
                endTimeStamp = ""
                docCredentials = doc[0]
                docScore = doc[1]
                matchedStr = re.match("SRT_(.*?.srt)_(.*?)_(.*?)_", docCredentials)
                if matchedStr != None:
                    fileName = matchedStr.group(1)
                    startTimeStamp = matchedStr.group(2)
                    endTimeStamp = matchedStr.group(3)
                    docTuple = (int(startTimeStamp), int(endTimeStamp), fileName)
                    for doc in self.docToTime:
                        if docTuple in self.docToTime[doc]:
                            print doc
                            break

    #shuffles the labeled sentences for training
    def shuffleSentences(self, tagged_sentences):
        shuffle(tagged_sentences)
        return tagged_sentences

    #trains doc2vec
    def trainDoc2Vec(self, tagged_sentences):
        #shuffle(tagged_sentences)
        self.doc2vecModel = gensim.models.doc2vec.Doc2Vec(alpha = 0.025, min_alpha = 0.025)#, min_count=1, window=10, size=100, workers=8)
        #print tagged_sentences
        self.doc2vecModel.build_vocab(tagged_sentences)
        #print "Tagged Sentences:"
        #print tagged_sentences
        for epoch in range(10):
            self.doc2vecModel.train(self.shuffleSentences(tagged_sentences), total_examples=self.doc2vecModel.corpus_count, epochs=self.doc2vecModel.iter)
            self.doc2vecModel.alpha -= 0.002  # decrease the learning rate
            self.doc2vecModel.min_alpha = self.doc2vecModel.alpha  # fix the learning rate, no decay

    #returns the most similar document list based on given string query
    def getMostSimilarDocs(self, query):
        queryTokens = query.split()
        queryDocVec = self.doc2vecModel.infer_vector(queryTokens)
        return self.doc2vecModel.docvecs.most_similar([queryDocVec])

    #TODO: deal with loading docToTime
    def loadModel(self, modelName = 'Doc2VecModel_SRT'):
        self.doc2vecModel = gensim.models.doc2vec.Doc2Vec.load('wordVecModel')

    def saveModel(self, fName = 'Doc2VecModel_SRT'):
        if self.doc2vecModel is None:
            print 'Error! Doc2Vec has not been initialized.'
        else:
            self.doc2vecModel.save(fName)

def main():
    parser = doc2vec()
    parser.saveModel()

if __name__ == "__main__":
    main()
