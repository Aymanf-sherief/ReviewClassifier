import math as m

import numpy as np
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
        self.negative_num = float(self.negative['X1'].size())
        self.positive_num = float(self.positive['X1'].size())

        self.data['positive'] = self.positive['X1']
        self.data['negative'] = self.negative['X1']

        # reader = sf.SFrame.read_csv('Reviews.csv')
        # self.pos_data = reader[reader['Score'] > 3]
        # self.neg_data = reader[reader['Score'] < 3]
        # self.neg_data, self.neg_test = self.neg_data.random_split(.9, seed=70)
        # self.pos_data, self.pos_test = self.pos_data.random_split(.99, seed=71)
        # self.positive_num = float(self.pos_data['Text'].size())
        # self.negative_num = float(self.neg_data['Text'].size())

        self.data, self.test_data = self.data.random_split(.8, seed=4)
        print self.negative_num
        print self.positive_num

    def word_count(self, Type, X):
        X = set(X.split())

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
        word_set = set(review.split())
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
        word_set = set(review.split())
        negativeP = np.longdouble(1.0)
        positiveP = np.longdouble(1.0)

        for word in self.positive_count:
            if word in word_set:
                positiveP *= float(self.positive_count[word])
                positiveP = self.round_up(positiveP, 4)
            else:
                positiveP *= 1.0 - self.positive_count[word]
                positiveP = self.round_up(positiveP, 4)
                # print positiveP
        for word in self.negative_count:
            if word in word_set:
                negativeP *= float(self.negative_count[word])
                negativeP = self.round_up(negativeP, 4)
            else:
                negativeP *= 1.0 - self.negative_count[word]
                negativeP = self.round_up(negativeP, 4)
        # print negativeP
        positiveP *= self.positive_num / (self.positive_num + self.negative_num)
        negativeP *= self.negative_num / (self.positive_num + self.negative_num)
        ptest = positiveP / (positiveP * negativeP)
        ntest = negativeP / (positiveP * negativeP)
        # print ptest
        # print ntest

        if ptest > ntest:

            # print review + "\n is a positive review with probability: " + str(positiveP/negativeP)
            return 1
        else:
            # print review + "\n is a negative review with probability: " + str(negativeP/negativeP)
            return 0

    def round_up(self, x, place):
        return round(x + 5 * 10 ** (-1 * (place + 1)), place)
