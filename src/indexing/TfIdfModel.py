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

#calcTFidf(self.corpus): Training
#self.calcTFidfQuery(query) : returns response, tfidf vector of query. response is the queryTfidfVec
#calcCosineSim(queryTfidfVec, tfidfMatrix, top_n) : returns top_n similar docs given queryTFidfVec from above, and tfidf Matrix in training
#printTopChoices(topChoices): prints given topCHoices, returned from calcCosineSim
class TfIdf:
    def __init__(self, corpus = None):
        self.tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, max_df = 0.9, stop_words = 'english')
        self.tfidf_matrix = None
        self.corpus = corpus
        if corpus != None:
            self.calcTFidf(corpus)
    #Todo: get highest ranked words of each document, do tfidf on srt
    #Todo: tfidf on the query itself(to limit search on words)
    #idf gives  the inverse of number of times word appears in other documents, get words with high idf
    #words with just high tf
    #phrase queries with tfidf to match highest score....
    def testQuery(self, query):
        response = self.calcTFidfQuery(query)
        topChoices = self.calcCosineSim(response)
        self.printTopChoices(topChoices)

    def calcCosineSim(self, queryTfidfVec, top_n = 10):
        if self.tfidf_matrix is None:
            print 'Error! tfidfMatrix has not yet been initialized.'
        else:
            cosine_similarities = linear_kernel(queryTfidfVec, self.tfidf_matrix).flatten()
            related_docs_indices = [i for i in cosine_similarities.argsort()[::-1]]
            return [(index, cosine_similarities[index]) for index in related_docs_indices][0:top_n]

    def calcTFidfQuery(self, query):
        response = self.tfidf.transform([query])
        feature_names = self.tfidf.get_feature_names()
        print "Query"
        #print response #response- sk learn matrix: (doc_num, feature_name): tfidf score
        #print feature_names
        for col in response.nonzero()[1]:
            print feature_names[col], ', ', response[0, col]
        return response

    def calcTFidf(self, corpus):
        self.tfidf_matrix =  self.tfidf.fit_transform(corpus)
        feature_names = self.tfidf.get_feature_names()
        dense = self.tfidf_matrix.todense()
        stopWordsSet = self.tfidf.get_stop_words() #the stopWordsSet


        #TODO: figure out this part, what is this doiong?
        currDoc = dense[1].tolist()[0] #filter out to get doc (i+1)
        phrase_scores = [pair for pair in zip(range(0, len(currDoc)), currDoc) if pair[1] > 0] #pair of featurename to feature score
        sorted_phrase_scores = sorted(phrase_scores, key=lambda t: t[1] * -1)
        print phrase_scores
        print "Sorted scores"
        print sorted_phrase_scores
        for scoreTuple in sorted_phrase_scores:
            phraseToScore = str(feature_names[scoreTuple[0]]) + ', ' + str(scoreTuple[1])
            print phraseToScore

        #for phrase, score in [(feature_names[word_id], score) for (word_id, score) in sorted_phrase_scores][:20]:
        #    print('{0: <20} {1}'.format(phrase, score))

    def printTopChoices(self, topChoices):
        print "TFIDF:"
        print topChoices
        for choice in topChoices:
            print self.corpus[choice[0]]

def main():
    parser = TfIdf()

if __name__ == "__main__":
    main()
