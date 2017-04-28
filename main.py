import NaiveClassifier as nv

nc = nv.NaiveClassifier('negative-data.csv', 'positive-data.csv')
nc.log_train()
e = 0.0
i = 0.0
for tweet in nc.test_data['positive']:
    if nc.classify(tweet) != 1:
        e += 1.0
print "positive test error: "
print e / float(nc.test_data['positive'].size())
e = 0.0
for tweet in nc.test_data['negative']:
    if nc.classify(tweet) != 0:
        e += 1.0
print "negative test error: "
print e / float(nc.test_data['negative'].size())
txt = ""
while txt != "exit":
    txt = raw_input()
    print nc.classify(txt.split())
#
