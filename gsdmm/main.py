# -*- coding: utf-8 -*-
from datetime import datetime
import pickle
import codecs
from vocab.ldaVocabulary import LDAVocabulary
import numpy as np
from gsdmm.mgp import MovieGroupProcess
import io


if __name__ == "__main__":
    customDictionary = set()
    category = "food"
    with codecs.open("E:\\projects\\AiProductDescWriter\\server_data\\%s\\data\\customDictionary.txt" % category, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            customDictionary.add(line.strip())

    stopWords = set()
    with codecs.open("E:\\libs\\dphanlp\\1.6.4\\data\\dictionary\\stopwords.txt", 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            stopWords.add(line.strip())
    voca = LDAVocabulary(stopwords=stopWords, customDictionary=customDictionary, customDictionaryOnly=False)

    docs = set()
    with io.open("E:\\projects\\AiProductDescWriter\\server_data\\%s\\data\\mergeResult" % category, 'r',
                 encoding='utf-8') as fin:
        for line in fin.readlines():
            arr = line.split('\t')
            if len(arr[2])>0:
                docs.add(arr[1].strip())
    print(docs)

    stList = []
    for doc in docs:
        if len(doc) > 2:
            doc = np.array(voca.doc_to_ids(doc=doc, training=True), dtype=object)
            stList.append(doc)
    print(len(stList))
    print("vocab size ", len(voca.vocabs))
    stList = voca.cut_low_freq(stList, 3)
    print("vocab size after cut ", len(voca.vocabs))

    mgp = MovieGroupProcess(K=300, n_iters=200, alpha=0.5, beta=0.01, V=len(voca.vocabs))
    y = mgp.fit(stList)
    with io.open('testResult.txt', 'w+', encoding='utf-8') as fout:
        yMap = {}
        for dz, st in zip(y, docs):
            if dz in yMap:
                yMap[dz].append(st)
            else:
                yMap[dz] = [st]

        for key, sts in yMap.items():
            fout.write("###############%d#############\n" % key)
            for st in sts:
                fout.write(st + '\n')
            fout.write('\n\n')
