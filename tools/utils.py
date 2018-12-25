# -*- coding: utf-8 -*-
from lda import LDA
import io
import numpy as np

def loadStopWords(stopword_file):
    pass

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

def loadLDAModel(word_topic_file):
    num_topics = word_topic_file.split("/")[-1].split(".")[0].split("_")[-1]
    topic_word = [[] for _ in range(num_topics)]
    topic_sum = [0] * num_topics
    with io.open(word_topic_file, 'r', encoding='utf-8') as f:
        for line in f:
            cols = line.strip().split()
            word_id = int(cols[0])
            for index in range(1, len(cols)):
                topic_id, cnt = [int(item) for item in cols[index].split(':')]
                topic_word[topic_id].append((word_id, cnt))
                topic_sum[topic_id] += cnt
    return topic_word, topic_sum

def dumpTopicWords(outPath, lda, vocab, topk):
    d = lda.worddist()
    d = d * np.log(len(lda.docs) * 1.0 / np.asarray(vocab.docfreq))
    with io.open(outPath, 'w+', encoding='utf-8') as fout:
        for i in range(lda.K):
            ind = np.argpartition(d[i], -topk)[-topk:]  # an array with the indexes of the 10 words with the highest probabilitity in the topic
            fout.write('topic %d\n' % i)
            for j in ind:
                fout.write(vocab[j] + '\n')
            fout.write('\n')