# -*- coding: utf-8 -*-
import sys
from datetime import datetime
import pickle
import codecs
import ch_vocabulary_sentenceLayer
from lda_sentenceLayer import lda_gibbs_sampling1
import numpy as np

if __name__ == "__main__":
    customDictionary = set()
    with codecs.open("E:\\projects\\AiProductDescWriter\\server_data\\food\\data\\customDictionary.txt", 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            customDictionary.add(line.strip())

    stopWords = set()
    with codecs.open("E:\\libs\\dphanlp\\1.6.4\\data\\dictionary\\stopwords.txt", 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            stopWords.add(line.strip())
    voca = ch_vocabulary_sentenceLayer.CHVocabularySentenceLayer(stopwords=stopWords, customDictionary=customDictionary, customDictionaryOnly=False)

    with codecs.open("E:\\projects\\AiProductDescWriter\\server_data\\food\\data\\mergeResult", 'r', encoding='utf-8') as fin:
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
        if len(value)>2:
            str = ','.join(value)
            doc = np.array(voca.doc_to_ids(doc=str, training=True), dtype=object)
            docs.append(doc)
    print(len(docs))
    print("vocab size ", len(voca.vocas))
    docs = voca.cut_low_freq(docs, 3)

    np.random.shuffle(docs)
    trainNum = int(len(docs)*0.9)
    trainDocs = docs[:trainNum]
    testDocs = docs[trainNum:]

    st = datetime.now()
    iterations, scores = 250, []
    topicNum = 100
    lda = lda_gibbs_sampling1(K=topicNum, alpha=0.01, beta=0.5, docs=trainDocs, V=voca.size())
    perpl, cnt, ar, nmi, p, r, f = [], 0, [], [], [], [], []

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
                with codecs.open(('lda.%dtopics.pkl' % (int(sys.argv[1]))), 'wb+') as out:
                    pickle.dump({'lda':lda, 'vocab': voca}, out)
            perpl.append(features[0])

            noImproveStepNum += 1
            if noImproveStepNum>3:
                break

    d = lda.worddist()
    with codecs.open('topicWords.txt', 'w+', encoding='utf-8') as fout:
        for i in range(topicNum):
            ind = np.argpartition(d[i], -15)[-15:] # an array with the indexes of the 10 words with the highest probabilitity in the topic
            fout.write('topic %d\n' % i)
            for j in ind:
                fout.write(voca[j] + '\n')
            fout.write('\n')

    print("It finished. Total time:", datetime.now() - st)
