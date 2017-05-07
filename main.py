import NaiveClassifier as nv

nc = nv.NaiveClassifier('negative-data.csv', 'positive-data.csv')
nc.log_train()  # train the classifier
e = 0.0  # error
n = 0.0  # neutral
for review in nc.data['positive']:  # test performance on positive data
    c = nc.classify(review)
    if c == -1:
        e += 1.0
    elif c == 0:
        n += 1.0
print "positive error: "
print e / float(nc.data['positive'].size())
print "positive neutrals: "
print n / float(nc.data['positive'].size())
e = 0.0
n = 0.0
for review in nc.data['negative']:  # test performance on negative data
    c = nc.classify(review)
    if c == 1:
        e += 1.0
    elif c == 0:
        n += 1.0
print "negative error: "
print e / float(nc.data['negative'].size())
print "negative neutrals: "
print n / float(nc.data['positive'].size())

txt = ""
while txt != "exit":  # while input isn't exit: classify input
    txt = raw_input()
    print nc.classify(txt.split())

