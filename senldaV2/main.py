# -*- coding: utf-8 -*-
import sys
from datetime import datetime
import pickle
import codecs
from vocab import senLDAVocabulary
from senLDAV2 import SenLDAV2
import numpy as np
from tools.utils import dumpTopicWords

if __name__ == "__main__":
    customDictionary = set()
    category = "food"
    with codecs.open("E:\\projects\\AiProductDescWriter\\server_data\\%s\\data\\customDictionary.txt" % category, 'r',
                     encoding='utf-8') as fin:
        for line in fin.readlines():
            customDictionary.add(line.strip())

    stopWords = set()
    with codecs.open("E:\\libs\\dphanlp\\1.6.4\\data\\dictionary\\stopwords.txt", 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            stopWords.add(line.strip())
    voca = senLDAVocabulary.SenLDAVocabulary(stopwords=stopWords, customDictionary=customDictionary, customDictionaryOnly=False)

    with codecs.open("E:\\projects\\AiProductDescWriter\\server_data\\%s\\data\\mergeResult" % category, 'r',
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
        if len(value) > 2:
            str = ','.join(value)
            doc = np.array(voca.doc_to_ids(doc=str, training=True), dtype=object)
            docs.append(doc)
    print(len(docs))
    print("vocab size ", len(voca.vocabs))
    docs = voca.cut_low_freq(docs, 3)
    print("vocab size after cut ", len(voca.vocabs))

    voca.dump_vocabulary('../model/%sSenLDAV2Vocabulary.txt' % category)

    np.random.shuffle(docs)
    trainNum = int(len(docs) * 0.9)
    trainDocs = docs[:trainNum]
    testDocs = docs[trainNum:]


    st = datetime.now()
    iterations, scores = 500, []
    stsTopicNum = 50
    topicNum = 500
    alpha = 0.6433 / stsTopicNum
    beta = 1.64*1e-4 * stsTopicNum + 1.4528713 / topicNum
    gamma = 5.276*1e-5 * stsTopicNum + 0.2156 / topicNum
    print(alpha, beta, gamma)
    lda = SenLDAV2(SK=stsTopicNum, K=topicNum, alpha=alpha, beta=beta, gamma=gamma, docs=trainDocs, V=voca.size())
    perpl, cnt, ar, nmi, p, r, f = [], 0, [], [], [], [], []

    voca.segmentor = None
    minValPerpl = 100000000
    minIter = 0
    noImproveStepNum = 0
    for i in range(iterations):
        starting = datetime.now()
        print("iteration:", i, )
        lda.inference()
        print("Took:", datetime.now() - starting)
        if i % 5 == 0:
            print ("Iteration:", i, "Perplexity:", lda.perplexity())
            features = lda.heldOutPerplexity(testDocs, 3)
            print("Held-out:", features[0])
            if features[0] < minValPerpl:
                minValPerpl = features[0]
                minIter = i
                noImproveStepNum = 0
                print("Iteration:", i, "min perplexity:", minValPerpl)
                with codecs.open(('../model/senLDAV2.%dtopics.pkl' % topicNum), 'wb+') as out:
                    pickle.dump({'lda':lda, 'vocab': voca}, out)
            perpl.append(features[0])

            noImproveStepNum += 1
            if noImproveStepNum>100:
                break

    dumpTopicWords('topicWords.txt', lda, voca, 15)

    print("It finished. Total time:", datetime.now() - st)
