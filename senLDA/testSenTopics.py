# -*- coding: utf-8 -*-
import jieba
import pickle
import numpy as np
import io

if __name__ == '__main__':
    topicNum = 25
    fh = io.open('lda.25topics.alpha0.0100002.pkl', 'rb')
    mDict = pickle.load(fh)
    lda = mDict['lda']
    vocab = mDict['vocab']

    for word in vocab.vocas_id.keys():
        jieba.add_word(word)

    testDoc = '目前正是新鲜柚子即将上市的时间，保证新鲜，清新爽口，现摘现发，皮薄肉多，叶酸天堂，含多种维生素c，对高血压，心脑血管及肾脏，最佳的食疗水果，因此红心蜜柚也是孕期佳品'
    stsTopics = []
    for sts in testDoc.split('，'):
        topicDist = np.zeros(topicNum)
        for word in jieba.cut(sts):
            Id = -1
            if word in vocab.vocas_id:
                Id = vocab.vocas_id[word]
            else:
                continue
            topicDist = topicDist + lda.n_z_t[:, Id]
        stsTopic = np.argmax(topicDist)
        stsTopics.append(stsTopic)
    print(stsTopics)

    fh.close()