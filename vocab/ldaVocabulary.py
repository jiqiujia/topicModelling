# -*- coding: utf-8 -*-
import jieba
import numpy as np
import re
import jieba.posseg as pseg
from vocab.my_vocabulary import Vocabulary

class LDAVocabulary(Vocabulary):
    def __init__(self, stopwords, customDictionary, customDictionaryOnly=False):
        Vocabulary.__init__(self, stopwords, customDictionary, customDictionaryOnly)

    def is_stopword(self, w):
        return w in self.stopwords

    def doc_to_ids(self, doc, training=True):
        l = []
        words = dict()
        for term in pseg.cut(doc):
            id = self.term_to_id(term, training)
            if id is not None:
                l.append(id)
                if not id in words:
                    words[id] = 1
                    self.docfreq[id] += 1 # It counts in how many documents a word appears. If it appears in only a few, remove it from the vocabulary using cut_low_freq()
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

        def conv(doc):
            new_doc = []
            for id in doc:
                if id in conv_map: new_doc.append(conv_map[id])
            return new_doc
        return [conv(doc) for doc in corpus]


    def __getitem__(self, v):
        return self.vocabs[v]

    def size(self):
        return len(self.vocabs)
