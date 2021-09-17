def Classify(featues_train, labels_train):

    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(featues_train, labels_train)
    return gnb