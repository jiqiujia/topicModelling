# -*- coding: utf-8 -*-
import pickle
import numpy as np
import io

from testSenLDATopics import getDocTopicFromWords
from tools.hanlpsegment import HanlpStandardTokenizer

if __name__ == '__main__':
    topicNum = 500
    fh = io.open('../model/senLDAV2.%dtopics.pkl' % topicNum, 'rb')
    mDict = pickle.load(fh)
    fh.close()
    lda = mDict['lda']
    vocab = mDict['vocab']
    print(len(vocab.vocabs))

    vocab.segmentor = HanlpStandardTokenizer("-Djava.class.path=.;../hanlp-1.7.1.jar;E:/dlprojects/topicModelling")
    vocab.segmentor.add_custom_words(vocab.customDictionary)
    print(len(vocab.vocabs))

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
    rawDocs = id2StsMap.values()
    docs = vocab.convert_docs2id(rawDocs)
    final_z_m_s = lda.predict(docs)

    sz2Doc = {}
    for z_s, doc in zip(final_z_m_s, rawDocs):
        for sz, sts in zip(z_s, doc):
            if sz in sz2Doc:
                sz2Doc[sz].append(sts)
            else:
                sz2Doc[sz] = [sts]
    with io.open('senLDAV2StsResult.txt', 'w+', encoding='utf-8') as fout:
        for sz, stsList in sz2Doc.items():
            fout.write("################%d##################\n" % sz)
            for sts in stsList:
                fout.write(sts+'\n')
            fout.write('\n\n')
