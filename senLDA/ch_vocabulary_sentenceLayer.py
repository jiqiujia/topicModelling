# -*- coding: utf-8 -*-
import jieba
import numpy as np
import re
import jieba.posseg as pseg

class CHVocabularySentenceLayer:
    def __init__(self, stopwords, customDictionary, customDictionaryOnly=False):
        self.vocas = []        # id to word
        self.vocas_id = dict() # word to id
        self.docfreq = []      # id to document frequency
        self.customDictionary = customDictionary
        self.customDictionaryOnly = customDictionaryOnly
        self.stopwords = stopwords
        for word in customDictionary:
            jieba.add_word(word)

    def is_stopword(self, w):
        return w in self.stopwords

    def term_to_id(self, term, training):
        word = term.word
        if self.is_stopword(word): return None
        if self.customDictionaryOnly:
            if word not in self.customDictionary: return None
        else:
            if not (term.flag[0]=='n' or term.flag[0]=='v' or term.flag[0]=='a'):
                return None
        try:
            term_id = self.vocas_id[word]
        except:
            if not training: return None
            term_id = len(self.vocas)
            self.vocas_id[word] = term_id
            self.vocas.append(word)
            self.docfreq.append(0)
        return term_id

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
        self.vocas_id = dict()
        conv_map = dict()
        for id, term in enumerate(self.vocas):
            freq = self.docfreq[id]
            if freq > threshold:
                new_id = len(new_vocas)
                self.vocas_id[term] = new_id
                new_vocas.append(term)
                new_docfreq.append(freq)
                conv_map[id] = new_id
        self.vocas = new_vocas
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


    def __getitem__(self, v):
        return self.vocas[v]

    def size(self):
        return len(self.vocas)
