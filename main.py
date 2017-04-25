import NaiveClassifier as nv

nc = nv.NaiveClassifier('negative-data.csv', 'positive-data.csv')
nc.train()
e = 0.0
i = 0.0
for tweet in nc.test_data['positive']:
    if nc.classify(tweet) != 1:
        e += 1.0
print e / float(nc.test_data['positive'].size())

while (True):
    txt = raw_input()
    print nc.classify(txt)
