import decimal
import math as m

import nltk
import sframe as sf


class NaiveClassifier:
    negative = sf.SFrame()
    positive = sf.SFrame()
    data = sf.SFrame()
    negative_count = {}
    positive_count = {}
    negative_num = 0.0
    positive_num = 0.0
    test_data = sf.SFrame()

    def __init__(self, file1, file2):
        print 'reading...'

        self.negative = sf.SFrame.read_csv("negative-data.csv", header=False)
        self.positive = sf.SFrame.read_csv("positive-data.csv", header=False)

        self.data['positive'] = self.positive['X1']
        self.data['negative'] = self.negative['X1']
        self.data['positive'] = self.data['positive'].apply(self.filter_words)
        self.data['negative'] = self.data['negative'].apply(self.filter_words)
        print self.data['positive'][-1]
        # reader = sf.SFrame.read_csv('Reviews.csv')
        # self.pos_data = reader[reader['Score'] > 3]
        # self.neg_data = reader[reader['Score'] < 3]
        # self.neg_data, self.neg_test = self.neg_data.random_split(.9, seed=70)
        # self.pos_data, self.pos_test = self.pos_data.random_split(.99, seed=71)
        # self.positive_num = float(self.pos_data['Text'].size())
        # self.negative_num = float(self.neg_data['Text'].size())

        self.data, self.test_data = self.data.random_split(.6, seed=0)
        self.negative_num = float(self.data['negative'].size())
        self.positive_num = float(self.data['positive'].size())

        print self.negative_num
        print self.positive_num

    def word_count(self, Type, X):
        X = set(X)

        if Type == 'negative':
            for word in X:
                if word not in self.positive_count:
                    self.positive_count[word] = 0.0
                if word in self.negative_count:
                    self.negative_count[word] += 1.0
                else:
                    self.negative_count[word] = 1.0
        elif Type == 'positive':
            for word in X:
                if word not in self.negative_count:
                    self.negative_count[word] = 0.0
                if word in self.positive_count:
                    self.positive_count[word] += 1.0
                else:
                    self.positive_count[word] = 1.0

    def prob_word_count(self):
        for word in self.positive_count:
            self.positive_count[word] /= self.positive_num

        for word in self.negative_count:
            self.negative_count[word] /= self.negative_num

    def log_word_count(self):
        for word in self.positive_count:
            if self.positive_count[word] != 0.0:
                self.positive_count[word] = m.log(self.positive_count[word]) - m.log(self.positive_num)
            else:
                self.positive_count[word] = - m.log(self.positive_num)

        for word in self.negative_count:
            if self.negative_count[word] != 0.0:
                self.negative_count[word] = m.log(self.negative_count[word]) - m.log(self.negative_num)
            else:
                self.negative_count[word] = - m.log(self.negative_num)

    def filter_counts(self):
        to_remove = []
        for word in self.positive_count:
            if (self.positive_count[word] == 1.0 and self.negative_count[word] == 1.0) or (
                            self.positive_count[word] == 0.0 and self.negative_count[word] == 0.0):
                to_remove.append(word)
        print to_remove

    def log_train(self):
        for review in self.data['negative']:
            self.word_count('negative', review)
        for review in self.data['positive']:
            self.word_count('positive', review)

        self.log_word_count()
        self.using_log = True
        # self.filter_counts()
        # print self.positive_count
        # print self.negative_count
        print 'Training done successfully...'

    def prob_train(self):
        for review in self.data['negative']:
            self.word_count('negative', review)
        for review in self.data['positive']:
            self.word_count('positive', review)
        self.prob_word_count()
        self.using_log = False
        # self.filter_counts()
        # print self.positive_count
        # print self.negative_count
        print 'Training done successfully...'

    def classify(self, review):
        if self.using_log:
            return self.log_classify(review)
        else:
            return self.prob_classify(review)

    def log_classify(self, review):
        word_set = set(review)
        negativeP = 0.0
        positiveP = 0.0

        for word in self.positive_count:
            if word in word_set:
                positiveP += float(self.positive_count[word])
            else:
                positiveP += m.log(1.0 - m.exp(self.positive_count[word]))
        for word in self.negative_count:
            if word in word_set:
                negativeP += float(self.negative_count[word])
            else:
                negativeP += m.log(1.0 - m.exp(self.negative_count[word]))

        positiveP += m.log(self.positive_num / (self.positive_num + self.negative_num))
        negativeP += m.log(self.negative_num / (self.positive_num + self.negative_num))
        ptest = positiveP - (positiveP + negativeP)
        ntest = negativeP - (positiveP + negativeP)
        # print ptest
        # print ntest

        if ptest > ntest:

            # print review + "\n is a positive review with probability: " + str(positiveP/negativeP)
            return 1
        else:
            # print review + "\n is a negative review with probability: " + str(negativeP/negativeP)
            return 0

    def prob_classify(self, review):
        word_set = set(review)
        decimal.getcontext().prec = 8000

        negativeP = decimal.Decimal(1.0)
        positiveP = decimal.Decimal(1.0)

        for word in word_set:
            if word in self.positive_count:
                positiveP *= decimal.Decimal(float(self.positive_count[word]))

            if word in self.negative_count:
                negativeP *= decimal.Decimal(float(self.negative_count[word]))


        positiveP *= decimal.Decimal(self.positive_num / (self.positive_num + self.negative_num))
        negativeP *= decimal.Decimal(self.negative_num / (self.positive_num + self.negative_num))
        try:
            ptest = positiveP / decimal.Decimal(positiveP + negativeP)
            ntest = negativeP / decimal.Decimal(positiveP + negativeP)
        except:
            print positiveP
            print negativeP
            ptest = 1
            ntest = 1
        # print ptest
        # print ntest

        if ptest > ntest:

            # print review + "\n is a positive review with probability: " + str(positiveP/negativeP)
            return 0
        else:
            # print review + "\n is a negative review with probability: " + str(negativeP/negativeP)
            return 1

    def round_up(self, x, place):
        return round(x + 5 * 10 ** (-1 * (place + 1)), place)

    def filter_words(self, line):
        # nltk.download()
        is_noun = lambda pos: pos[:2] == 'NN' or pos[:2] == 'NNS' or pos[:2] == 'NNP'
        is_adverb = lambda pos: pos[:2] == 'RB' or pos[:2] == 'RBR' or pos[:2] == 'RBS'
        is_adj = lambda pos: pos[:2] == 'JJ' or pos[:2] == 'JJR' or pos[:2] == 'JJS'
        is_verb = lambda pos: pos[:2] == 'VB' or pos[:2] == 'VBG' or pos[:2] == 'VBD' or \
                              pos[:2] == 'VBP' or pos[:2] == 'VBZ' or pos[:2] == 'VBN'
        included = lambda pos: is_noun(pos) or is_adj(pos) or is_adverb(pos) or is_verb(pos)

        # do the nlp stuff
        tokenized = nltk.word_tokenize(line)
        words = [word for (word, pos) in nltk.pos_tag(tokenized) if included(pos)]

        return words
