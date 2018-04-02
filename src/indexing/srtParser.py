# things to do with wordvectors: https://radimrehurek.com/gensim/models/keyedvectors.html
# model['computer']  # raw numpy vector of a word, with trained word2vec
#includes genism for phrase2vec
#frequency of words for the whole file --> could be in the list for stop Words
#import youtube api
#porter stemming: for the whole corpus before indexing https://pythonprogramming.net/stemming-nltk-tutorial/ ***
#stop-wrds ****
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


#phrase2vec, doc2vec (4)
#get closest word, if not in word, then go to next closest (5)
#stopwords: english online (1) -Done
#tfidf to get the highest ranked words, and also get stopwords through dissimilar videos srt (2) https://stevenloria.com/tf-idf/
#tfidf query (noun phrases,etc) (3)
#training
#ranking of the video snippets indexed by the user through survey
#tfidf of 1 minute time slots (6)
#Porter Stemmer with unique words(7)

#remove words with count of 1
#find snippet of when this explanation started
#doc2vec for query and word and word2vec for word queries(word vectors addition)
#testing: look at course for potential queries and good time slots
#tfidf: check about scikit matrix and vector for tfidf[each row represents document, and each column represents the term]
#tfidf for stop words [done]
#k-means in the future for clustering [based on theme]
#doc2vec: https://rare-technologies.com/doc2vec-tutorial/
#training: list of words, and label (label can be just number of document)
#exact word similarity is fine for query similarity


# TODO: Could the issue be with stemming for not getting back the queries, stem both query and result!!!!!? not stemmed
#TODO: train on larger set of data like the syllabus, and then add additional training for each document, read up on doc2vec
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
from DocVecModel import DocVec
from WordVecModel import WordVec
from TfIdfModel import TfIdf


'''
def buildSrtCorpus(self):
    lectures = defaultdict(list)
    with open("data/import/sentences.csv", "r") as sentences_file:
        reader = csv.reader(sentences_file, delimiter=',')
        reader.next()
        for row in reader:
            episodes[row[1]].append(row[4])

    for episode_id, text in episodes.iteritems():
        episodes[episode_id] = "".join(text)

corpus = []
for id, episode in sorted(episodes.iteritems(), key=lambda t: int(t[0])):
    corpus.append(episode)
#put in openFile
'''
#Had the same issue, should be added through feature_names = tf.get_feature_names() I think. Thanks for the walkthrough, quite fun.
#feature_names: phrases or n-grams number
'''
tfidf_matrix =  tf.fit_transform(corpus)
feature_names = tf.get_feature_names()
episode = dense[0].tolist()[0] #filter out to get episode 1 pharses
phrase_scores = [pair for pair in zip(range(0, len(episode)), episode) if pair[1] > 0]
'''
#sorted(phrase_scores, key=lambda t: t[1] * -1)[:5] gives you the sorted score
'''
#phrase with scores
sorted_phrase_scores = sorted(phrase_scores, key=lambda t: t[1] * -1)
for phrase, score in [(feature_names[word_id], score) for (word_id, score) in sorted_phrase_scores][:20]:
   print('{0: <20} {1}'.format(phrase, score))
'''

'''
#phrases that don't appear in other episodes
with open("data/import/tfidf_scikit.csv", "w") as file:
    writer = csv.writer(file, delimiter=",")
    writer.writerow(["EpisodeId", "Phrase", "Score"])

    doc_id = 0
    for doc in tfidf_matrix.todense():
        print "Document %d" %(doc_id)
        word_id = 0
        for score in doc.tolist()[0]:
            if score > 0:
                word = feature_names[word_id]
                writer.writerow([doc_id+1, word.encode("utf-8"), score])
            word_id +=1
        doc_id +=1
'''
'''
class LabeledLineSentence(object):
    def __init__(self, filename):
        self.filename = filename
    def __iter__(self):
        for srtID, line in enumerate(open(filename)):
            yield LabeledSentence(words=line.split(), labels=['SRT_%s' % srtID])
'''
class parseSrt:
    def __init__(self):
        self.listFiles = ['closed_caption.srt']
        self.wordToTime = {}
        self.vecToTime = {}
        self.text = []

        self.doc2vecModel = None
        self.wordvecModel = None
        self.tfidf = None#TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, max_df = 0.9, stop_words = 'english')

        self.labeledSentences = []
        self.new_model = None
        self.alphanum = re.compile('[^a-zA-Z0-9]')
        self.p = PorterStemmer()
        self.uniqueVocab = set()
        self.stop_words = set(self.setupData('stopWords.txt')).union(set(stopwords.words('english')))
        self.path = './srtFiles'
        self.documentLen = 4 #sets docLen to 4 lines
        self.corpus = []
        self.docToTime = {}


    def setupData(self, fileName):
        contents = []
        f = open(fileName)
        for line in f:
            contents.append(line.decode('utf-8'))
        f.close()
        result = ('\n'.join(contents)).split()
        return result

    #Todo: get highest ranked words of each document, do tfidf on srt FORM_FIELD_TO_SALESFORCE_FIELD
    #Todo: tfidf on the query itself(to limit search on words)
    #idf gives  the inverse of number of times word appears in other documents, get words with high idf
    #words with just high tf
    #phrase queries with tfidf to match highest score....
    def calcCosineSim(self, queryTfidfVec, tfidfMatrix, top_n = 10):
        cosine_similarities = linear_kernel(queryTfidfVec, tfidfMatrix).flatten()
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

    def calcTFidf(self):
        tfidf_matrix =  self.tfidf.fit_transform(self.corpus)
        feature_names = self.tfidf.get_feature_names()
        dense = tfidf_matrix.todense()
        stopWordsSet = self.tfidf.get_stop_words()
        print 'Stop Words'
        print stopWordsSet

        currDoc = dense[1].tolist()[0] #filter out to get doc (i+1)
        phrase_scores = [pair for pair in zip(range(0, len(currDoc)), currDoc) if pair[1] > 0] #pair of featurename to feature score
        #print feature_names
        sorted_phrase_scores = sorted(phrase_scores, key=lambda t: t[1] * -1)
        print phrase_scores
        print "Sorted scores"
        print sorted_phrase_scores
        for scoreTuple in sorted_phrase_scores:
            phraseToScore = str(feature_names[scoreTuple[0]]) + ', ' + str(scoreTuple[1])
            print phraseToScore

        #for phrase, score in [(feature_names[word_id], score) for (word_id, score) in sorted_phrase_scores][:20]:
        #    print('{0: <20} {1}'.format(phrase, score))
        print len(feature_names)
        print len(currDoc)
        #print phrase_scores
        response = self.calcTFidfQuery("base case")
        topChoices = self.calcCosineSim(response, tfidf_matrix)
        print "TFIDF:"
        print topChoices
        for choice in topChoices:
            print self.corpus[choice[0]]

    def getRandomQuery(self):
        word = 'recursion'#random.choice(list(self.uniqueVocab))
        if self.wordvecModel is None:
            self.wordvecModel = gensim.models.Word2Vec.load('wordVecModel')
        print 'Time slots'
        print self.wordToTime[word]
        print self.vecToTime[tuple(self.wordvecModel[word])]
        similarList = self.wordvecModel.most_similar(positive=[word], topn=10)
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
        print 'Word2vec test'
        wordList = ['recursion', 'variable', 'for loop' , 'classes', 'network', 'condition', 'function', 'computer science', 'arithmetic']
        #print(model.most_similar(positive=['computer', 'science'], negative=[], topn=5)) #concatenates and prints out top 5 similar words [useful for study guides]

        for word in wordList:
            phrase = word.split()
            phraseList = [w.strip() for w in phrase]

            similarList = self.wordvecModel.most_similar(positive = phraseList, negative = [], topn=5)
            print word
            print similarList

    def openFile(self):
        # TODO looking at data: deal with stop words, deal with commas
        #TODO: just copy this code over to train all the files
        #doc2vec tags as keywords from trained syllabus or word2vec trained stuff or tfidf term
    # for subdir, dirs, files in os.walk(rootdir):
    #     for file in files:
    #         filepath = subdir + os.sep + file
    #
    #         if filepath.endswith(".asm"):
    #             print (filepath)
        print 'Training'
        for subdir, directories, files in os.walk(self.path):
            for fileName in files:
                fileName = subdir + os.sep + fileName
                #print fileName
                if '.DS_Store' not in fileName and not os.path.isdir(fileName):
                    #fileName = self.path + fileName
                     #print fileName
                #for fileName in self.listFiles:
                    print fileName
                    subs = pysubs2.load(fileName)
                    leftDoc = self.documentLen
                    currDoc = []
                    currDocTimeStamp = [0 ,0]
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
                        line = [str(xx).translate(None, string.punctuation) for xx in line]
                        line = [xx for xx in line if xx not in self.stop_words]

                        if leftDoc == 4:
                            currDocTimeStamp[0] = startTimeStamp
                        currDoc += line #" ".join(line) + ' '
                        leftDoc -= 1

                        if leftDoc == 0 or i == len(subs) - 1:
                            #print currDoc
                            currDocTimeStamp[1] = endTimeStamp
                            currLabeledSentence = LabeledSentence(currDoc,['SRT_%s_%s_%s_%d' %(fileName, currDocTimeStamp[0], currDocTimeStamp[1],  i)])#[currDoc,['SRT_%s_%d' %(fileName, i)] ] #(words = currDoc, labels = )
                            self.labeledSentences.append(currLabeledSentence)

                            currDocStr = " ".join(currDoc)

                            timeDocList = []
                            if currDocStr in self.docToTime:
                                timeDocList = self.docToTime.get(currDocStr)
                            timeDocList.append((currDocTimeStamp[0], currDocTimeStamp[1], fileName))
                            self.docToTime[currDocStr] = timeDocList

                            self.corpus.append(currDocStr)
                            leftDoc = self.documentLen
                            currDoc = []
                            currDocTimeStamp = [-1,-1]


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

        self.doc2vecModel = DocVec(self.labeledSentences, self.docToTime)
        self.wordvecModel = WordVec(self.text)
        self.tfidf = TfIdf(self.corpus)
        print "Training DONE"

        # self.calcTFidf()
        # #print "Self.Labeled"
        # #print self.labeledSentences
        # self.trainDoc2Vec(self.labeledSentences)
        # print "Doc2Vec"
        # query = "recursion"
        # similarDocs =  self.getMostSimilarDocs(query)
        #
        # #print similarDocs
        # self.printSimilarDocs(similarDocs)

    def printSimilarDocs(self, simDocList):
        #print self.docToTime['So saw looked algorithms things like exhaustive enumeration We said well searching answer search space carefully one time']
        #print simDocList[0]
        #print 'Similar Docs List:'
        #print self.docToTime
        #print simDocList
        #print self.doc2vecModel.docvecs.most_similar('SRT_./srtFiles/6wTuOMgTrU4.srt_2583870_2598760_595')
        for doc in simDocList:
            fileName = ""
            startTimeStamp = ""
            endTimeStamp = ""
            docCredentials = doc[0]
            docScore = doc[1]
            matchedStr = re.match("SRT_(.*?.srt)_(.*?)_(.*?)_", docCredentials)
            #print doc
            if matchedStr != None:
                fileName = matchedStr.group(1)
                startTimeStamp = matchedStr.group(2)
                endTimeStamp = matchedStr.group(3)
                docTuple = (int(startTimeStamp), int(endTimeStamp), fileName)
                #print docTuple
                for doc in self.docToTime:
                    if docTuple in self.docToTime[doc]:
                        print doc
                        break


    def buildPreTrainedVec(self):
        preTrainedModel = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
        preTrainedModel.save('srtModel')

    def buildWordVec(self):
        #print self.text
        #preTrainedModel = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
        #preTrainedModel.build_vocab(self.text, update=True)

        model = gensim.models.Word2Vec(self.text, min_count=1)
        model.save('wordVecModel')
        #model.reset_from(preTrainedModel)
        #preTrainedModel.train(self.text)
        #preTrainedModel.save('srtModel') #model for word2vec from syllabus
        print("Done saving wordVec")

    def loadWordVec(self):
        self.wordvecModel = gensim.models.Word2Vec.load('wordVecModel')
        for word in self.wordToTime:
            self.vecToTime[tuple(self.wordvecModel[word])] = self.wordToTime[word]

    def shuffleSentences(self, tagged_sentences):
        shuffle(tagged_sentences)
        return tagged_sentences

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

    def getMostSimilarDocs(self, query):
        queryTokens = query.split()
        queryDocVec = self.doc2vecModel.infer_vector(queryTokens)
        return self.doc2vecModel.docvecs.most_similar([queryDocVec])

    def getMostSimilarList(self, query):
        queryTokens = query.split()

    #use this method to test the three different queries
    def testQueries(self):
        wordList = ['recursion', 'network stack', 'data structure', 'algorithms']
        for word in wordList:
            print 'Testing TFIDF:'
            self.tfidf.testQuery(word)

            print 'Testing Doc2Vec:'
            self.doc2vecModel.testQuery(word)

            print 'Testing Word2Vec:'
            self.wordvecModel.testQuery(word)


def main():
    parser = parseSrt()
    parser.openFile()
    parser.testQueries()
    #parser.buildWordVec()
    #parser.loadWordVec()
    #parser.getRandomQuery()

if __name__ == "__main__":
    main()
