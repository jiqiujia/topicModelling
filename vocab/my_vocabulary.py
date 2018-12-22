# -*- coding: utf-8 -*-
import jieba
import numpy as np
import re
import jieba.posseg as pseg
import io


class Vocabulary:

    def __init__(self, stopwords=None, customDictionary=None, customDictionaryOnly=False):
        self.vocabs = []        # id to word
        self.vocab2id = dict() # word to id
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
            term_id = self.vocab2id[word]
        except:
            if not training: return None
            term_id = len(self.vocabs)
            self.vocab2id[word] = term_id
            self.vocabs.append(word)
            self.docfreq.append(0)
        return term_id

    def __getitem__(self, v):
        return self.vocabs[v]

    def size(self):
        return len(self.vocabs)

    def dump_vocabulary(self, path):
        with io.open(path, 'w+', encoding='utf-8') as fout:
            for id, (word, docFreq) in enumerate(zip(self.vocabs, self.docfreq)):
                fout.write('%d\t%s\t%d\n' % (id, word, docFreq))

    def load_vocabulary(self, path):
        with io.open(path, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.split('\t')
                self.vocab2id[int(line[0])] = line[1]


