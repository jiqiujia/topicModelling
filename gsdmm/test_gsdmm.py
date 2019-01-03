from unittest import TestCase
from gsdmm.mgp import MovieGroupProcess
import numpy
import io
from tools.hanlpsegment import HanlpStandardTokenizer


class TestGSDMM(TestCase):
    '''This class tests the Panel data structures needed to support the RSK model'''

    def setUp(self):
        numpy.random.seed(47)

    def tearDown(self):
        numpy.random.seed(None)

    def compute_V(self, texts):
        V = set()
        for text in texts:
            for word in text:
                V.add(word)
        return len(V)

    def test_grades(self):

        grades = list(map(list, [
            "A",
            "A",
            "A",
            "B",
            "B",
            "B",
            "B",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "D",
            "D",
            "F",
            "F",
            "P",
            "W"
        ]))

        grades = grades + grades + grades + grades + grades
        mgp = MovieGroupProcess(K=100, n_iters=100, alpha=0.001, beta=0.01)
        y = mgp.fit(grades, self.compute_V(grades))
        self.assertEqual(len(set(y)), 7)
        for words in mgp.cluster_word_distribution:
            self.assertTrue(len(words) in {0,1}, "More than one grade ended up in a cluster!")

    def test_short_text(self):
        # there is no perfect segmentation of this text data:
        texts = [
            "where the red dog lives",
            "red dog lives in the house",
            "blue cat eats mice",
            "monkeys hate cat but love trees",
            "green cat eats mice",
            "orange elephant never forgets",
            "orange elephant must forget",
            "monkeys eat banana",
            "monkeys live in trees",
            "elephant",
            "cat",
            "dog",
            "monkeys"
        ]

        texts = [text.split() for text in texts]
        V = self.compute_V(texts)
        mgp = MovieGroupProcess(K=30, n_iters=100, alpha=0.2, beta=0.01, V=V)
        y = mgp.fit(texts, V)
        self.assertTrue(len(set(y))<10)
        self.assertTrue(len(set(y))>3)

    def test_my_short_text(self):
        customDictionary = set()
        category = "food"
        with io.open("E:\\projects\\AiProductDescWriter\\server_data\\%s\\data\\customDictionary.txt" % category,
                         'r',
                         encoding='utf-8') as fin:
            for line in fin.readlines():
                customDictionary.add(line.strip())
        segmentor = HanlpStandardTokenizer("-Djava.class.path=.;../hanlp-1.7.1.jar;E:/dlprojects/topicModelling")
        segmentor.add_custom_words(customDictionary)

        stopWords = set()
        with io.open("E:\\libs\\dphanlp\\1.6.4\\data\\dictionary\\stopwords.txt", 'r', encoding='utf-8') as fin:
            for line in fin.readlines():
                stopWords.add(line.strip())

        docs = set()
        category = 'food'
        with io.open("E:\\projects\\AiProductDescWriter\\server_data\\%s\\data\\mergeResult" % category, 'r',
                         encoding='utf-8') as fin:
            for line in fin.readlines():
                docs.add(line.split('\t')[1].strip())

        stList = []
        for doc in docs:
            words = segmentor.cut(doc)
            words = [word for word in words if word not in stopWords]
            stList.append(words)

        V = self.compute_V(stList)
        mgp = MovieGroupProcess(K=50, n_iters=100, alpha=0.1, beta=0.01)
        y = mgp.fit(stList, V)