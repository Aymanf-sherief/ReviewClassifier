import NaiveClassifier as nv

nc = nv.NaiveClassifier('negative-data.csv', 'positive-data.csv')
nc.log_train()
e = 0.0
i = 0.0
for tweet in nc.data['positive']:
    if nc.classify(tweet) != 1:
        e += 1.0
print "positive error: "
print e / float(nc.data['positive'].size())
e = 0.0
for tweet in nc.data['negative']:
    if nc.classify(tweet) != 0:
        e += 1.0
print "negative error: "
print e / float(nc.data['negative'].size())
txt = ""
while txt != "exit":
    txt = raw_input()
    print nc.classify(txt.split())
#
