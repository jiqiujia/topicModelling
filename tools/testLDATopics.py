# -*- coding: utf-8 -*-
from tools.utils import loadLDAModel
from vocab.ldaVocabulary import LDAVocabulary
import jieba
import io
import pickle


if __name__ == '__main__':
    # word_topic_file = '../model/food_lda_100.model'
    # topic_word = loadLDAModel(word_topic_file)
    #
    # vocab = LDAVocabulary()
    # vocab.load_vocabulary('../model/ldaVocabulary.txt')
    topicNum = 100
    fh = io.open('../model/lda.%dtopics.pkl' % topicNum, 'rb')
    mDict = pickle.load(fh)
    lda = mDict['lda']
    vocab = mDict['vocab']
    print(len(vocab.vocabs))

    for word in vocab.vocab2id.keys():
        jieba.add_word(word)

    lda.dumpDocWordTopics('docWordTopics.txt', vocab)

    testDocs = ['目前正是新鲜柚子即将上市的时间，保证新鲜，清新爽口，现摘现发，皮薄肉多，叶酸天堂，含多种维生素c，对高血压，心脑血管及肾脏，最佳的食疗水果，因此红心蜜柚也是孕期佳品']

    testDocs = vocab.convert_docs2id(testDocs)
    res_z_m_n, res_n_m_z = lda.predict(testDocs)

    res = []
    for z_m_n, doc in zip(res_z_m_n, testDocs):
        for z, word in zip(z_m_n, doc):
            res.append('%s:%d' % (vocab.vocabs[word], z))
    print(" ".join(res))