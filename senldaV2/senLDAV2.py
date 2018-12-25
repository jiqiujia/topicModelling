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
        SenLDA.__init__(self)
        self.type = 'senLDAV2'
        self.K = K
        self.SK = SK    # sentence topic number
        self.alpha = alpha  # parameter of topics prior
        self.beta = beta  # parameter of words prior
        self.gamma = gamma # parameter of sts topics prior
        self.docs = docs  # a list of lists, each inner list contains the indexes of the words in a doc, e.g.: [[1,2,3],[2,3,5,8,7],[1, 5, 9, 10 ,2, 5]]
        self.S = 0
        for doc in docs:
            self.S += len(doc)
        self.V = V  # how many different words in the vocabulary i.e., the number of the features of the corpus
        # Definition of the counters
        self.z_m_s = []  # topic assignments for each of the S sentences in the corpus
        self.s_m_z = []
        self.n_s_z = np.zeros((self.SK, self.K))
        self.n_m_s = np.zeros((len(self.docs), SK))
        self.n_s = np.zeros(SK)     # overall number of words assigned to sentence topic s

        self.z_m_n = []     # topic assignments for each of the N words in the corpus. N: total number of words in the corpus (not the vocabulary size).
        self.n_m_z = np.zeros((len(self.docs), K),
                              dtype=np.float64) #+ alpha  # |docs|xK topics: number of sentences assigned to topic z in document m
        self.n_z_t = np.zeros((K, V),
                              dtype=np.float64) #+ beta  # (K topics) x |V| : number of times a word v is assigned to a topic z
        self.n_z = np.zeros(K) #+ V * beta  # (K,) : overall number of words assigned to a topic z
        for m, doc in enumerate(docs):  # Initialization of the data structures I need.
            z_ns = []
            z_s = []
            for sentence in doc:
                sz = np.random.randint(0, SK)
                z_s.append(sz)
                self.n_s[sz] += len(sentence)  #
                self.n_m_s[m, sz] += 1
                z_n = []
                for t in sentence:
                    z = np.random.randint(0, K)  # Randomly assign a topic to a sentence. Recall, topics have ids 0 ... K-1. randint: returns integers to [0,K[
                    z_n.append(z)  # Keep track of the topic assigned
                    self.n_s_z[sz, z] += 1
                    self.n_m_z[m, z] += 1  # increase the number of sentences assigned to topic z in the m doc.
                    self.n_z_t[z, t] += 1  # .... number of times a word is assigned to this particular topic
                    self.n_z[z] += len(sentence)  # increase the counter of words assigned to z topic
                z_ns.append(z_n)
            self.z_m_n.append(np.array(z_ns))  # update the array that keeps track of the topic assignements in the words of the corpus.
            self.z_m_s.append(np.array(z_s))

    def get_sts_full_conditional(self, sentence, m, sz, z_n, n_m_s):
        prod_nom, prod_den = [], []  # numerator, denominator
        zdict = Counter(z_n)
        for z, znum in zdict.items():
            self.n_s_z[sz, z] -= znum
            for x in range(znum):
                prod_nom.append(self.gamma + self.n_s_z[:, z] + x)
        prod_nom = np.array(prod_nom, dtype=np.float64)

        left_denominator = self.n_s + self.beta * self.V
        for x in range(len(sentence)):
            quantity = left_denominator + x
            prod_den.append(quantity)

        prod_den = np.array(prod_den, dtype=np.float64)
        prodall1 = np.divide(prod_nom, prod_den)
        prodall = np.prod(prodall1, axis=0)

        right = (n_m_s[m, :] + self.alpha)
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
                    z = z_n[n]
                    n_m_z[z] -= 1
                    self.n_z_t[z, t] -= 1  # Decrease the number of the words assigned to topic z
                    self.n_z[z] -= 1  # Decrease the total number of words assigned to topic z
                    self.n_s_z[sz, z] -= 1
                    p_z = ((self.n_z_t[:, t] + self.beta) * (self.n_s_z[:, z] + self.gamma) / (self.n_z + self.V * self.beta))
                    p_z = p_z / p_z.sum()
                    new_z = np.random.multinomial(1, p_z).argmax()  # One multinomial draw, for a distribution over topics, with probabilities summing to 1, return the index of the topic selected.
                    # set z the new topic and increment counters
                    z_n[n] = new_z
                    n_m_z[new_z] += 1
                    self.n_z_t[new_z, t] += 1
                    self.n_z[new_z] += 1
                    self.n_s_z[sz, new_z] += 1
                # 先对词采样，再对句子采样，不确定这样子实现对不对
                n_m_s[sz] -= 1
                self.n_s -= len(sentence)
                p_sz = self.get_sts_full_conditional(sentence, m, sz, z_n, n_m_s)
                new_sz = np.random.multinomial(1, p_sz).argmax()
                n_m_s[new_sz] += len(sentence)
                self.n_s[new_sz] += 1

    def perplexity(self, docs=None):
        if docs == None: docs = self.docs
        phi = self.worddist()
        log_per = 0
        N = 0
        for m, doc in enumerate(docs):
            theta = self.n_m_s[m] / (len(doc) + self.SK * self.alpha)
            for sen in doc:
                for w in sen:
                    log_per -= np.log(np.dot(np.dot(theta, self.n_s_z / (self.SK + self.K * self.gamma)), phi[:, w]))
                N += len(sen)
        return np.exp(log_per / N)

    def heldOutPerplexity(self, docs, iterations):
        N, log_per, z_m_n = 0, 0, []
        n_m_z1, n_z_t, n_z = (np.zeros((len(docs), self.K)) + self.alpha), (
                np.zeros((self.K, self.V)) + self.beta), np.zeros(self.K)
        for m, doc in enumerate(docs):  # Initialization of the data structures I need.
            z_n = []
            for sentence in doc:
                z = np.random.randint(0,
                                      self.K)  # Randomly assign a topic to a sentence. Recall, topics have ids 0 ... K-1. randint: returns integers to [0,K[
                z_n.append(z)  # Keep track of the topic assigned
                n_m_z1[m, z] += 1  # increase the number of sentences assigned to topic z in the m doc.
                n_z_t[z, sentence.astype(
                    dtype=np.int32)] += 1  # .... number of times a word is assigned to this particular topic
                n_z[z] += len(sentence)  # increase the counter of words assigned to z topic
            z_m_n.append(np.array(
                z_n))  # update the array that keeps track of the topic assignements in the sentences of the corpus.
        for i in range(iterations):
            for m, doc in enumerate(docs):
                z_n, n_m_z = z_m_n[m], n_m_z1[m]
                for sid, sentence in enumerate(doc):
                    z = z_n[sid]  # Obtain the topic that was assigned to sentences
                    n_m_z[z] -= 1  # Decrease the number of the sentences in the current document assigned to topic z
                    n_z_t[
                        z, sentence.astype(dtype=np.int32)] -= 1  # Decrease the number of the words assigned to topic z
                    n_z[z] -= len(sentence)  # Decrease the total number of words assigned to topic z
                    p_z = self.get_full_conditional(sentence, m, z, n_z, n_m_z1)
                    new_z = np.random.multinomial(1, p_z).argmax()
                    z_n[sid] = new_z
                    n_m_z[new_z] += 1
                    n_z_t[new_z, sentence.astype(dtype=np.int32)] += 1
                    n_z[new_z] += len(sentence)
        phi = self.worddist()
        Kalpha = self.K * self.alpha
        for m, doc in enumerate(docs):
            theta = n_m_z1[m] / (len(doc) + Kalpha)
            for sen in doc:
                for w in sen:
                    log_per -= np.log(np.inner(phi[:, w], theta))
                N += len(sen)
        topicDist = n_m_z1 / n_m_z1.sum(axis=1)[:, np.newaxis]
        return np.exp(log_per / N), topicDist