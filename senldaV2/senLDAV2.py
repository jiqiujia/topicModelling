import numpy as np, codecs, json, sys
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter
import warnings
import io

warnings.filterwarnings("error")
try:
    import cPickle as pickle
except:
    import pickle
from senlda.senLDA import SenLDA


class SenLDAV2(SenLDA):
    def __init__(self, SK=25, K=100, alpha=0.5, beta=0.5, gamma=0.5, docs=None, V=None):
        SenLDA.__init__(self, K, alpha, beta, docs, V)
        self.type = 'senLDAV2'
        self.K = K
        self.SK = SK  # sentence topic number
        self.alpha = alpha  # parameter of topics prior
        self.beta = beta  # parameter of words prior
        self.gamma = gamma  # parameter of sts topics prior
        self.docs = docs  # a list of lists, each inner list contains the indexes of the words in a doc, e.g.: [[1,2,3],[2,3,5,8,7],[1, 5, 9, 10 ,2, 5]]
        self.S = 0
        for doc in docs:
            self.S += len(doc)
        self.V = V  # how many different words in the vocabulary i.e., the number of the features of the corpus
        # Definition of the counters
        self.z_m_s = []  # topic assignments for each of the S sentences in the corpus
        self.n_s_z = np.zeros((self.SK, self.K), dtype=np.float64)
        self.n_m_s = np.zeros((len(self.docs), self.SK), dtype=np.float64)
        self.n_s = np.zeros(self.SK)  # overall number of words assigned to sentence topic s

        self.z_m_n = []  # topic assignments for each of the N words in the corpus: list of list
        self.n_m_z = np.zeros((len(self.docs), self.K),
                              dtype=np.float64)  # + alpha  # |docs|xK topics: number of sentences assigned to topic z in document m
        self.n_z_t = np.zeros((self.K, self.V),
                              dtype=np.float64)  # + beta  # (K topics) x |V| : number of times a word v is assigned to a topic z
        self.n_z = np.zeros(self.K)  # + V * beta  # (K,) : overall number of words assigned to a topic z
        for m, doc in enumerate(self.docs):  # Initialization of the data structures I need.
            z_ns = []
            z_s = []
            for sentence in doc:
                sz = np.random.randint(0, self.SK)
                z_s.append(sz)
                self.n_s[sz] += len(sentence)  #
                self.n_m_s[m, sz] += 1
                z_n = []
                for t in sentence:
                    z = np.random.randint(0,
                                          self.K)  # Randomly assign a topic to a sentence. Recall, topics have ids 0 ... K-1. randint: returns integers to [0,K[
                    z_n.append(z)  # Keep track of the topic assigned
                    self.n_s_z[sz, z] += 1
                    self.n_m_z[m, z] += 1  # increase the number of sentences assigned to topic z in the m doc.
                    self.n_z_t[z, t] += 1  # .... number of times a word is assigned to this particular topic
                    self.n_z[z] += 1  # increase the counter of words assigned to z topic
                z_ns.append(z_n)
            self.z_m_n.append(np.array(
                z_ns))  # update the array that keeps track of the topic assignements in the words of the corpus.
            self.z_m_s.append(np.array(z_s))

    def get_sts_full_conditional(self, sentence, m, zdict, n_m_s):
        prod_nom, prod_den = [], []  # numerator, denominator
        for z, znum in zdict.items():
            for x in range(znum):
                prod_nom.append(self.gamma + self.n_s_z[:, z] + x)
        prod_nom = np.array(prod_nom, dtype=np.float64)

        left_denominator = self.n_s + self.beta * self.K
        for x in range(len(sentence)):
            quantity = left_denominator + x
            prod_den.append(quantity)

        prod_den = np.array(prod_den, dtype=np.float64)
        prodall1 = np.divide(prod_nom, prod_den)
        prodall = np.prod(prodall1, axis=0)

        right = (n_m_s + self.alpha)
        p_z = prodall * right
        try:
            p_z /= np.sum(p_z)
        except:
            print(m, len(sentence), sentence)
            print(p_z)
            print(right, n_m_s.shape)
            print(prodall1.shape, prodall1)
            print(prodall.shape, prodall)
            assert False
        return p_z

    def inference(self):
        """    The learning process. Here only one iteration over the data.
               A loop will be calling this function for as many iterations as needed.     """
        for m, doc in enumerate(self.docs):
            z_s, n_m_s = self.z_m_s[m], self.n_m_s[m]
            z_n, n_m_z = self.z_m_n[m], self.n_m_z[m]  # Take the topics of the sentences and the number of sentences assigned to each topic
            for sid, sentence in enumerate(doc):
                sz = z_s[sid]  # Obtain the topic that was assigned to sentences
                for n, t in enumerate(sentence):
                    z = z_n[sid][n]
                    n_m_z[z] -= 1
                    self.n_z_t[z, t] -= 1  # Decrease the number of the words assigned to topic z
                    self.n_z[z] -= 1  # Decrease the total number of words assigned to topic z
                    self.n_s_z[sz, z] -= 1
                    p_z = ((self.n_z_t[:, t] + self.beta) * (self.n_s_z[sz, :] + self.gamma) / (
                                self.n_z + self.V * self.beta))
                    # print(np.sum(self.n_s_z<0))
                    # print(np.sum(self.n_z_t<0))
                    # print(np.sum(self.n_z)<0)
                    p_z = p_z / p_z.sum()
                    new_z = np.random.multinomial(1, p_z).argmax()  # One multinomial draw, for a distribution over topics, with probabilities summing to 1, return the index of the topic selected.
                    # set z the new topic and increment counters
                    z_n[sid][n] = new_z
                    n_m_z[new_z] += 1
                    self.n_z_t[new_z, t] += 1
                    self.n_z[new_z] += 1
                    self.n_s_z[sz, new_z] += 1
                # 先对词采样，再对句子采样，不确定这样子实现对不对
                n_m_s[sz] -= 1
                self.n_s[sz] -= len(sentence)
                zdict = Counter(z_n[sid])
                for z, znum in zdict.items():
                    self.n_s_z[sz, z] -= znum
                p_sz = self.get_sts_full_conditional(sentence, m, zdict, n_m_s)
                try:
                    new_sz = np.random.multinomial(1, p_sz).argmax()
                except:
                    print(np.sum(self.n_s_z<0))
                    print(np.sum(self.n_s<0))
                    print(np.sum(n_m_s<0))
                    assert False
                for z, znum in zdict.items():
                    self.n_s_z[new_sz, z] += znum
                z_s[sid] = new_sz
                n_m_s[new_sz] += 1
                self.n_s[new_sz] += len(sentence)

    def perplexity(self, docs=None, n_m_z=None, n_z_t=None, n_z=None):
        if docs is None: docs = self.docs
        if n_m_z is None: n_m_z = self.n_m_z
        if n_z is None: n_z = self.n_z
        phi = self.worddist(n_z_t, n_z)
        log_per = 0
        N = 0
        for m, doc in enumerate(docs):
            theta = self.n_m_s[m] / (len(doc) + self.SK * self.alpha)
            for sen in doc:
                for w in sen:
                    log_per -= np.log(np.dot(np.dot(theta, self.n_s_z / (self.SK + self.K * self.gamma)), phi[:, w]) + 1e-9)
                N += len(sen)
        return np.exp(log_per / N)

    def heldOutPerplexity(self, docs, iterations):
        z_m_s = []  # topic assignments for each of the S sentences in the corpus
        n_m_s = np.zeros((len(docs), self.SK))
        n_s = np.zeros(self.SK)  # overall number of words assigned to sentence topic s

        z_m_n = []  # topic assignments for each of the N words in the corpus. N: total number of words in the corpus (not the vocabulary size).
        n_m_z = np.zeros((len(docs), self.K),
                              dtype=np.float64)  # + alpha  # |docs|xK topics: number of sentences assigned to topic z in document m

        n_z = np.zeros(self.K)  # + V * beta  # (K,) : overall number of words assigned to a topic z
        for m, doc in enumerate(docs):  # Initialization of the data structures I need.
            z_ns = []
            z_s = []
            for sentence in doc:
                sz = np.random.randint(0, self.SK)
                z_s.append(sz)
                n_s[sz] += len(sentence)  #
                n_m_s[m, sz] += 1
                z_n = []
                for t in sentence:
                    z = np.random.randint(0,
                                          self.K)  # Randomly assign a topic to a sentence. Recall, topics have ids 0 ... K-1. randint: returns integers to [0,K[
                    z_n.append(z)  # Keep track of the topic assigned
                    n_m_z[m, z] += 1  # increase the number of sentences assigned to topic z in the m doc.
                    n_z[z] += len(sentence)  # increase the counter of words assigned to z topic
                z_ns.append(z_n)
            z_m_n.append(np.array(
                z_ns))  # update the array that keeps track of the topic assignements in the words of the corpus.
            z_m_s.append(np.array(z_s))
        for i in range(iterations):
            for m, doc in enumerate(docs):
                z_s, n_m_s1 = z_m_s[m], n_m_s[m]
                z_n, n_m_z1 = z_m_n[m], n_m_z[m]  # Take the topics of the sentences and the number of sentences assigned to each topic
                for sid, sentence in enumerate(doc):
                    sz = z_s[sid]  # Obtain the topic that was assigned to sentences
                    for n, t in enumerate(sentence):
                        z = z_n[sid][n]
                        n_m_z1[z] -= 1
                        n_z[z] -= 1  # Decrease the total number of words assigned to topic z
                        p_z = ((self.n_z_t[:, t] + self.beta) * (self.n_s_z[sz, :] + self.gamma) / (
                                n_z + self.V * self.beta))
                        try:
                            p_z = p_z / p_z.sum()
                        except:
                            print(p_z)
                            print(n_z)
                            assert False
                        # One multinomial draw, for a distribution over topics, with probabilities summing to 1, return the index of the topic selected.
                        new_z = np.random.multinomial(1, p_z).argmax()
                        # set z the new topic and increment counters
                        z_n[sid][n] = new_z
                        n_m_z1[new_z] += 1
                        n_z[new_z] += 1
                    # 先对词采样，再对句子采样
                    n_m_s1[sz] -= 1
                    n_s[sz] -= len(sentence)
                    prod_nom, prod_den = [], []  # numerator, denominator
                    zdict = Counter(z_n[sid])
                    for z, znum in zdict.items():
                        # self.n_s_z[sz, z] -= znum
                        for x in range(znum):
                            prod_nom.append(self.gamma + self.n_s_z[:, z] + x)
                    prod_nom = np.array(prod_nom, dtype=np.float64)

                    left_denominator = n_s + self.beta * self.K
                    for x in range(len(sentence)):
                        quantity = left_denominator + x
                        prod_den.append(quantity)
                    prod_den = np.array(prod_den, dtype=np.float64)
                    prodall1 = np.divide(prod_nom, prod_den)
                    prodall = np.prod(prodall1, axis=0)

                    right = (n_m_s1 + self.alpha)
                    p_sz = prodall * right
                    try:
                        p_sz /= np.sum(p_sz)
                        new_sz = np.random.multinomial(1, p_sz).argmax()
                    except:
                        print(i, m, len(sentence), sentence)
                        print('p_sz:', p_sz)
                        print('right:', right, n_m_s1.shape)
                        print(prodall1.shape, prodall1)
                        print(prodall.shape, prodall)
                        assert False
                    # for z, znum in zdict.items():
                    #     self.n_s_z[new_sz, z] += znum
                    z_s[sid] = new_sz
                    n_m_s1[new_sz] += 1
                    n_s[new_sz] += len(sentence)
        phi = self.worddist(self.n_z_t, n_z)
        log_per = 0
        N = 0
        for m, doc in enumerate(docs):
            theta = n_m_s[m] / (len(doc) + self.SK * self.alpha)
            for sen in doc:
                for w in sen:
                    log_per -= np.log(np.dot(np.dot(theta, self.n_s_z / (self.SK + self.K * self.gamma)), phi[:, w]) + 1e-9)
                N += len(sen)
        topicDist = n_m_z / n_m_z.sum(axis=1)[:, np.newaxis]
        return np.exp(log_per / N), topicDist


    def dumpDocWordTopics(self, outpath, vocab):
        with io.open(outpath, 'w+', encoding='utf-8') as fout:
            for z_n, doc in zip(self.z_m_n, self.docs):
                doc = [w for sublist in doc for w in sublist]
                zs = [z for sublist in z_n for z in sublist]
                line = ['%s:%d' % (vocab.vocabs[wordId], z) for z, wordId in zip(zs, doc)]
                fout.write(' '.join(line) + '\n')

    def subroutine(self, docs, z_m_s, n_m_s, z_m_n, n_m_z, n_z, n_s):
        for m, doc in enumerate(docs):
            z_s, n_m_s1 = z_m_s[m], n_m_s[m]
            z_n, n_m_z1 = z_m_n[m], n_m_z[m]  # Take the topics of the sentences and the number of sentences assigned to each topic
            for sid, sentence in enumerate(doc):
                sz = z_s[sid]  # Obtain the topic that was assigned to sentences
                if len(sentence)==0:
                    continue
                for n, t in enumerate(sentence):
                    z = z_n[sid][n]
                    n_m_z1[z] -= 1
                    n_z[z] -= 1  # Decrease the total number of words assigned to topic z
                    p_z = (np.reshape((self.n_z_t[:, t] + self.beta), -1) * (self.n_s_z[sz, :] + self.gamma) / (
                            n_z + self.V * self.beta))
                    try:
                        p_z = p_z / p_z.sum()
                    except:
                        print(p_z)
                        print(n_z)
                        assert False
                    # One multinomial draw, for a distribution over topics, with probabilities summing to 1, return the index of the topic selected.
                    new_z = np.random.multinomial(1, p_z).argmax()
                    # set z the new topic and increment counters
                    z_n[sid][n] = new_z
                    n_m_z1[new_z] += 1
                    n_z[new_z] += 1
                # 先对词采样，再对句子采样
                n_m_s1[sz] -= 1
                n_s[sz] -= len(sentence)
                prod_nom, prod_den = [], []  # numerator, denominator
                zdict = Counter(z_n[sid])
                for z, znum in zdict.items():
                    # self.n_s_z[sz, z] -= znum
                    for x in range(znum):
                        prod_nom.append(self.gamma + self.n_s_z[:, z] + x)
                prod_nom = np.array(prod_nom, dtype=np.float64)

                left_denominator = n_s + self.beta * self.K
                for x in range(len(sentence)):
                    quantity = left_denominator + x
                    prod_den.append(quantity)
                prod_den = np.array(prod_den, dtype=np.float64)
                prodall1 = np.divide(prod_nom, prod_den)
                prodall = np.prod(prodall1, axis=0)

                right = (n_m_s1 + self.alpha)
                p_sz = prodall * right
                try:
                    p_sz /= np.sum(p_sz)
                    new_sz = np.random.multinomial(1, p_sz).argmax()
                except:
                    print(m, len(sentence), sentence)
                    print('p_sz:', p_sz)
                    print('right:', right, n_m_s1.shape)
                    print(prodall1.shape, prodall1)
                    print(prodall.shape, prodall)
                    assert False
                # for z, znum in zdict.items():
                #     self.n_s_z[new_sz, z] += znum
                z_s[sid] = new_sz
                n_m_s1[new_sz] += 1
                n_s[new_sz] += len(sentence)
        return z_m_s
    def predict(self, docs):
        z_m_s = []  # topic assignments for each of the S sentences in the corpus
        n_m_s = np.zeros((len(docs), self.SK))
        n_s = np.zeros(self.SK)  # overall number of words assigned to sentence topic s

        z_m_n = []  # topic assignments for each of the N words in the corpus. N: total number of words in the corpus (not the vocabulary size).
        n_m_z = np.zeros((len(docs), self.K),
                         dtype=np.float64)  # + alpha  # |docs|xK topics: number of sentences assigned to topic z in document m

        n_z = np.zeros(self.K)  # + V * beta  # (K,) : overall number of words assigned to a topic z
        for m, doc in enumerate(docs):  # Initialization of the data structures I need.
            z_ns = []
            z_s = []
            for sentence in doc:
                sz = np.random.randint(0, self.SK)
                z_s.append(sz)
                n_s[sz] += len(sentence)  #
                n_m_s[m, sz] += 1
                z_n = []
                for t in sentence:
                    z = np.random.randint(0,
                                          self.K)  # Randomly assign a topic to a sentence. Recall, topics have ids 0 ... K-1. randint: returns integers to [0,K[
                    z_n.append(z)  # Keep track of the topic assigned
                    n_m_z[m, z] += 1  # increase the number of sentences assigned to topic z in the m doc.
                    n_z[z] += len(sentence)  # increase the counter of words assigned to z topic
                z_ns.append(z_n)
            z_m_n.append(np.array(
                z_ns))  # update the array that keeps track of the topic assignements in the words of the corpus.
            z_m_s.append(np.array(z_s))
        print("start burnin...")
        for i in range(200):
            self.subroutine(docs, z_m_s, n_m_s, z_m_n, n_m_z, n_z, n_s)
            if i % 10==0:
                print(i)
        z_m_s_list = []
        print("start sampling")
        for i in range(200):
            z_m_s = self.subroutine(docs, z_m_s, n_m_s, z_m_n, n_m_z, n_z, n_s)
            if i % 10 == 0:
                z_m_s_list.append(z_m_s)
                print(i)

        res_z_m_s = []
        for i, z_m_s in enumerate(z_m_s_list):
            for m, z_s in enumerate(z_m_s):
                if i==0:
                    res_z_m_s.append([z_s])
                else:
                    res_z_m_s[m].append(z_s)

        final_z_m_s = []
        for z_s_list in res_z_m_s:
            zs_matrix = np.asarray(z_s_list)
            z_s = []
            for i in range(zs_matrix.shape[1]):
                z_s.append(np.bincount(zs_matrix[:, i]).argmax())
            final_z_m_s.append(z_s)
        return final_z_m_s