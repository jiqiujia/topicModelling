# -*- coding: utf-8 -*-
import pickle
import numpy as np
import io

from hanlpsegment import HanlpStandardTokenizer

if __name__ == '__main__':
    topicNum = 100
    fh = io.open('../model/senLDA.%dtopics.pkl' % topicNum, 'rb')
    mDict = pickle.load(fh)
    fh.close()
    lda = mDict['lda']
    vocab = mDict['vocab']
    print(len(vocab.vocabs))

    vocab.segmentor = HanlpStandardTokenizer("-Djava.class.path=.;../hanlp-1.7.1.jar;E:/dlprojects/topicModelling")
    vocab.segmentor.add_custom_words(vocab.customDictionary)
    print(len(vocab.vocabs))

    d = lda.worddist()
    d = d * np.log(len(lda.docs) * 1.0 / np.asarray(vocab.docfreq))
    with io.open('topicWords.txt', 'w+', encoding='utf-8') as fout:
        for i in range(topicNum):
            ind = np.argpartition(d[i], -15)[
                  -15:]  # an array with the indexes of the 10 words with the highest probabilitity in the topic
            fout.write('topic %d\n' % i)
            for j in ind:
                fout.write(vocab[j] + '\n')
            fout.write('\n')
    lda.dumpDocWordTopics('senLDADocWordTopics%d.txt' % topicNum, vocab)

    testDoc = '目前正是新鲜柚子即将上市的时间，保证新鲜，清新爽口，现摘现发，皮薄肉多，叶酸天堂，含多种维生素c，对高血压，心脑血管及肾脏，最佳的食疗水果，因此红心蜜柚也是孕期佳品'
    stsTopics = []
    for sts in testDoc.split('，'):
        topicDist = np.zeros(topicNum)
        for word in vocab.segmentor.cut(sts):
            Id = -1
            if word in vocab.vocab2id:
                Id = vocab.vocab2id[word]
            else:
                continue
            topicDist = topicDist + lda.n_z_t[:, Id]
        stsTopic = np.argmax(topicDist)
        stsTopics.append((stsTopic, sts))
    print(stsTopics)
