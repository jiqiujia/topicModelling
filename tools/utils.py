# -*- coding: utf-8 -*-
from lda import LDA
import io

def dumpLDAModel(lda: LDA, outPath):
    word_topic_dict = {}
    for wi in range(lda.V):
        word_topic_dict[wi] = {}
        for k in range(lda.K):
            if lda.n_z_t[k][wi] > 0:
                word_topic_dict[wi][k] = lda.n_z_t[k][wi]
    with io.open(outPath, 'w+', encoding='utf-8') as fout:
        for word, topicDict in word_topic_dict.items():
            topics = []
            for topicId, cnt in topicDict.items():
                topics.append("{}:{}".format(topicId, int(cnt)))
            line = '{} {}\n'.format(word, ' '.join(topics))
            fout.write(line)


