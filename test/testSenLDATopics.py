# -*- coding: utf-8 -*-
import pickle
import numpy as np
import io

from hanlpsegment import HanlpStandardTokenizer

def getDocTopicFromWords(testDoc, n_z_t, vocab):
    stsTopics = []
    n_z_t = n_z_t / np.sum(n_z_t, axis=0, keepdims=True)
    for sts in testDoc.split('，'):
        topicDist = np.zeros(n_z_t.shape[0])
        for word in vocab.segmentor.cut(sts):
            if word in vocab.vocab2id:
                Id = vocab.vocab2id[word]
            else:
                continue
            topicDist = topicDist + n_z_t[:, Id]
        stsTopic = np.argmax(topicDist)
        val = topicDist[stsTopic]
        stsTopics.append((stsTopic, val, sts))
    print(stsTopics)
    return stsTopics

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

    category = 'food'
    with io.open("E:\\projects\\AiProductDescWriter\\server_data\\%s\\data\\mergeResult" % category, 'r',
                     encoding='utf-8') as fin:
        id2StsMap = {}
        for line in fin.readlines():
            line = line.split("\t")
            Id = line[0].split('_')[0]
            sts = line[1].strip()
            if Id in id2StsMap:
                id2StsMap[Id].append(sts)
            else:
                id2StsMap[Id] = [sts]

    print(len(id2StsMap))
    docs = []
    for key, value in id2StsMap.items():
        docs.append('，'.join(value))

    topic2StsMap = {}
    for doc in docs:
        for stsTopic in getDocTopicFromWords(doc, lda.n_z_t, vocab):
            topic, value, sts = stsTopic
            if topic in topic2StsMap:
                topic2StsMap[topic].add((sts, value))
            else:
                topic2StsMap[topic] = {(sts, value)}

    with io.open('senLDATestResult%d.txt' % topicNum, 'w+', encoding='utf-8')as fout:
        for topic, stsList in topic2StsMap.items():
            fout.write('##################topic %d###################\n' % topic)
            stsList = sorted(stsList, key=lambda x: x[1], reverse=True)
            for sts in stsList:
                fout.write('{}\t{:.3f}\n'.format(sts[0], sts[1]))
            fout.write('\n\n')



