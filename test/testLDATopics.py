# -*- coding: utf-8 -*-
from testSenLDATopics import getDocTopicFromWords
from tools.utils import loadLDAModel
from vocab.ldaVocabulary import LDAVocabulary
import io
import pickle
import numpy as np
from tools.utils import dumpTopicWords
from tools.topicMerger import TopicModelMerge
from tools.hanlpsegment import HanlpStandardTokenizer


if __name__ == '__main__':
    # word_topic_file = '../model/food_lda_100.model'
    # topic_word = loadLDAModel(word_topic_file)
    #
    # vocab = LDAVocabulary()
    # vocab.load_vocabulary('../model/ldaVocabulary.txt')
    category = 'food'

    topicNum = 1000
    fh = io.open('../model/lda.%dtopics.pkl' % topicNum, 'rb')
    mDict = pickle.load(fh)
    lda = mDict['lda']
    vocab = mDict['vocab']
    vocab.segmentor = HanlpStandardTokenizer("-Djava.class.path=.;../hanlp-1.7.1.jar;E:/dlprojects/topicModelling")
    vocab.segmentor.add_custom_words(vocab.customDictionary)
    print(len(vocab.vocabs))

    dumpTopicWords('ldaTopicWords%d.txt' % topicNum, lda, vocab, 15)
    lda.dumpDocWordTopics('docWordTopics%d.txt' % topicNum, vocab)

    topicMerger = TopicModelMerge(lda.K, '../model/{}_lda_{:d}.model'.format(category, lda.K))
    topicMap, topic_word = topicMerger.reduce_topic(15, 0.5, 1, '../model/merged_{}_lda_{:d}.model'.format(category, lda.K))
    lda.n_z_t = np.zeros((lda.K, lda.V))
    for k, wordCntList in enumerate(topic_word):
        for word, cnt in wordCntList:
            lda.n_z_t[k, word] = cnt

    lda.n_z = np.sum(lda.n_z_t, axis=1)
    lda.n_z = lda.n_z * 1.0 / np.sum(lda.n_z)

    for mi in range(len(lda.docs)):
        for wi in range(len(lda.z_m_n[mi])):
            if lda.z_m_n[mi][wi] in topicMap:
                lda.z_m_n[mi][wi] = topicMap[lda.z_m_n[mi][wi]]
    dumpTopicWords('mergedLDATopicWords%d.txt' % topicNum, lda, vocab, 15)


    rawTestDocs = ['目前正是新鲜柚子即将上市的时间，保证新鲜，清新爽口，现摘现发，皮薄肉多，叶酸天堂，含多种维生素c，对高血压，心脑血管及肾脏，最佳的食疗水果，因此红心蜜柚也是孕期佳品',]

    testDocs = vocab.convert_docs2id(rawTestDocs)
    res_z_m_n, res_n_m_z = lda.predict(testDocs)

    res = []
    for z_m_n, doc in zip(res_z_m_n, testDocs):
        for z, word in zip(z_m_n, doc):
            res.append('%s:%d' % (vocab.vocabs[word], z))
    print(" ".join(res))


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

    with io.open('ldaTestResult%d.txt' % topicNum, 'w+', encoding='utf-8')as fout:
        for topic, stsList in topic2StsMap.items():
            fout.write('##################topic %d###################\n' % topic)
            stsList = sorted(stsList, key=lambda x: x[1], reverse=True)
            for sts in stsList:
                fout.write('{}\t{:.3f}\n'.format(sts[0], sts[1]))
            fout.write('\n\n')
