import math as m

import nltk
import sframe as sf


class NaiveClassifier:
    negative = sf.SFrame()  # negative reviews
    positive = sf.SFrame()  # positive reviews
    data = sf.SFrame()  # contains both positive and negative data
    negative_count = {}  # count of negative words
    positive_count = {}  # count of positive words
    negative_num = 0.0  # num of negative reviews
    positive_num = 0.0  # num of positive reviews

    # test_data = sf.SFrame()  # test data - not always used

    def __init__(self, negative_file, positive_file):
        print 'reading...'
        # read negative and positive data
        self.negative = sf.SFrame.read_csv(negative_file, header=False)
        self.positive = sf.SFrame.read_csv(positive_file, header=False)
        # combine both sframes into data sframe in a manageable format
        self.data['positive'] = self.positive['X1'].apply(self.filter_words)  # applies filtering to training data
        self.data['negative'] = self.negative['X1'].apply(self.filter_words)
        # self.data, self.test_data = self.data.random_split(.6, seed=0)
        # extract num of positive and negative reviews
        self.negative_num = float(self.data['negative'].size())
        self.positive_num = float(self.data['positive'].size())
        print self.data
        print self.negative_num
        print self.positive_num

    def word_count(self, Type, X):  # counts word occurrences in a string and adds it to the correct counts dictionary
        X = set(X)  # take only distinct words
        if Type == 'negative':  # negative sentence
            for word in X:
                if word not in self.positive_count:  # if word not in positive counts then set its positive count to 0
                    self.positive_count[word] = 0.0
                if word in self.negative_count:  # if word in negative count then increment it else set it to 1
                    self.negative_count[word] += 1.0
                else:
                    self.negative_count[word] = 1.0
        elif Type == 'positive':
            for word in X:
                if word not in self.negative_count:  # if word not in negative counts then set its negative count to 0
                    self.negative_count[word] = 0.0
                if word in self.positive_count:  # if word in positive count then increment it else set it to 1
                    self.positive_count[word] += 1.0
                else:
                    self.positive_count[word] = 1.0

    def log_word_count(self):  # computes log probability of word counts
        for word in self.positive_count:
            if self.positive_count[word] != 0.0:  # can't take log of 0.0
                # log(a/b) = log a - log b

                self.positive_count[word] = m.log(self.positive_count[word]) - m.log(self.positive_num)
            else:
                self.positive_count[word] = - m.log(self.positive_num)
        for word in self.negative_count:  # do same for negative count
            if self.negative_count[word] != 0.0:
                self.negative_count[word] = m.log(self.negative_count[word]) - m.log(self.negative_num)
            else:
                self.negative_count[word] = - m.log(self.negative_num)

    def classify(self, review):  # takes an input review and decides whether it's positive, negative or neutral
        return self.log_classify(review)

    def log_classify(self, review):  # classifies using log word count
        word_set = set(review)  # take distinct words
        negativeP = 0.0
        positiveP = 0.0
        for word in self.positive_count:  # for each positive word we have
            if word in word_set:  # if word in input review
                positiveP += float(self.positive_count[word])  # sum it's log probability to total positive probability
            else:
                positiveP += m.log(1.0 - m.exp(self.positive_count[word]))  # else sum log (1-p) of word
        for word in self.negative_count:  # do same for each negative word
            if word in word_set:
                negativeP += float(self.negative_count[word])
            else:
                negativeP += m.log(1.0 - m.exp(self.negative_count[word]))
        # add effect of total positive vs negative probability
        positiveP += m.log(self.positive_num / (self.positive_num + self.negative_num))
        negativeP += m.log(self.negative_num / (self.positive_num + self.negative_num))
        # calculate total positive probability vs total negative probability of review
        ptest = positiveP - (positiveP + negativeP)
        ntest = negativeP - (positiveP + negativeP)
        # test variable is negative if review is negative, positive if review is positive, 0 if neutral by dafault
        test = (ptest - ntest) / (ptest + ntest)
        if test >= 0.005:  # add neutral threshold = 0.005
            # print review + "\n is a positive review with probability: " + str(positiveP/negativeP)
            return 1
        elif test <= -0.005:
            # print review + "\n is a negative review with probability: " + str(negativeP/negativeP)
            return -1
        else:
            return 0

    def log_train(self):  # trains the classifier on training data
        for review in self.data['negative']:  # count every positive review
            self.word_count('negative', review)
        for review in self.data['positive']:  # count every negative review
            self.word_count('positive', review)
        self.log_word_count()  # compute log probabilities
        print 'Training done successfully...'  # flag

    def filter_words(self, line):  # removes stop words and extracts useful words
        # nltk.download() -- for dependencies download
        is_noun = lambda pos: pos[:2] == 'NN' or pos[:2] == 'NNS' or pos[:2] == 'NNP'  # true if word is noun
        is_adverb = lambda pos: pos[:2] == 'RB' or pos[:2] == 'RBR' or pos[:2] == 'RBS'  # true if word is adverb
        is_adj = lambda pos: pos[:2] == 'JJ' or pos[:2] == 'JJR' or pos[:2] == 'JJS'  # true if word is adjective
        is_verb = lambda pos: pos[:2] == 'VB' or pos[:2] == 'VBG' or pos[:2] == 'VBD' or \
                              pos[:2] == 'VBP' or pos[:2] == 'VBZ' or pos[:2] == 'VBN'  # true if word is verb
        included = lambda pos: is_noun(pos) or is_adj(pos) or is_adverb(pos) or is_verb(pos)  # true if word is useful
        # do the nlp stuff
        tokenized = nltk.word_tokenize(line)  # tokenize review
        # create words list of useful words where included = true
        words = [word for (word, pos) in nltk.pos_tag(tokenized) if included(pos)
                 and word not in set(nltk.corpus.stopwords.words('english'))]
        return words
