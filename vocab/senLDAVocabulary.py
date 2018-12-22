# -*- coding: utf-8 -*-
import jieba
import numpy as np
import re
import jieba.posseg as pseg
from vocab.my_vocabulary import Vocabulary


class SenLDASentenceLayer(Vocabulary):
    def __init__(self, stopwords, customDictionary, customDictionaryOnly=False):
        Vocabulary.__init__(stopwords, customDictionary, customDictionaryOnly)

    def doc_to_ids(self, doc, training=True):
        l = []
        words = dict()
        doc_sents = re.split(r'~|，|？|。|！|；|,|\?|\.{2,}|!|;|:|：|\n|\r|——', doc)
        for sentence in doc_sents:
            miniArray = []
            for term in pseg.cut(sentence):
                id = self.term_to_id(term, training)
                if id != None:
                    miniArray.append(id)
                    if not id in words:
                        words[id] = 1
                        self.docfreq[id] += 1 # It counts in how many documents a word appears. If it appears in only a few, remove it from the vocabulary using cut_low_freq()
            l.append(np.array(miniArray, dtype=np.int32))
        return l

    def cut_low_freq(self, corpus, threshold=1):
        new_vocas = []
        new_docfreq = []
        self.vocab2id = dict()
        conv_map = dict()
        for id, term in enumerate(self.vocabs):
            freq = self.docfreq[id]
            if freq > threshold:
                new_id = len(new_vocas)
                self.vocab2id[term] = new_id
                new_vocas.append(term)
                new_docfreq.append(freq)
                conv_map[id] = new_id
        self.vocabs = new_vocas
        self.docfreq = new_docfreq
        return np.array([ self.conv(doc, conv_map) for doc in corpus])

    def conv(self, doc, conv_map, window=10):
        n = [np.array([conv_map[id] for id in sen if id in conv_map]) for sen in doc]
        n = [x for x in n if x.shape[0] > 0]
        m = []
        for x in n:
            if x.shape[0] > window:
                m.extend([x[i:i+window] for i in range(0, x.shape[0], window)])
            else:
                m.append(x)
        return  np.array(m)
