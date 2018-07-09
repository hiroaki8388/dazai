import os
from collections import defaultdict
from functools import reduce
from operator import add
import numpy as np

import gensim
from nltk import TextCollection
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.matutils import sparse2full

from snippets.reader import JapaneseCorpusReader


class JapaneseTextNormalizer(BaseEstimator, TransformerMixin):
    """Janomeで変換したwordの変換を行う

    """

    STOP_WORD_OF_POS = ['助詞', '助動詞', '記号']

    def __init__(self):
        pass

    def is_stopword(self, token, stop_word_of_pos):
        """特定の品詞に属するwordをstopwordと判定する"""
        part_of_speech = set(token.part_of_speech.split(','))

        return not part_of_speech.isdisjoint(stop_word_of_pos)

    def normalize(self, token, stop_word_pos):
        """janomeのTokenの集合を、原型の単語に変換する"""

        if not self.is_stopword(token, stop_word_pos):
            return token.base_form

    def fit(self, X, y=None):
        return self

    def transform(self, corpus, stop_word_pos=STOP_WORD_OF_POS):
        """
        corpusReaderのtokenからnormalizeした単語のlistを返す
        :param corpus: JapaneseCorpusReader
        :return: list(list(str))
        """
        transed = [list(filter(None, [self.normalize(word, stop_word_pos) for word in sent])) for sent in
                   corpus.sents()]

        return transed


class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    """
    scikit-learnにはword2vecのような変換が存在しないので、Gensimより作成する
    学習済みモデルはこちらのを使う
    http://aial.shiroyagi.co.jp/2017/02/japanese-word2vec-model-builder/
    """
    WORD_DIM = 50

    def __init__(self, path='../w2vmodel/latest-ja-word2vec-gensim-model/word2vec.gensim.model'):
        self.path = path
        self.id2word = None

        self._load()

    def _load(self):
        if os.path.exists(self.path):
            # 学習済みモデルのload
            self.id2word = gensim.models.Word2Vec.load(self.path)
        else:
            pass

    def fit(self, sent):
        """

        :param sent:
        :return:
        """

        return self

    def transform(self, sents):
        """
        現状Doc2Vecの学習済みモデルが存在しないので、word2vecで変換した単語単位のモデルの平均ベクトルを取る
        :param sents:
        :return:
        """
        for sent in sents:
            wordvecs = [self._word2vec(word) for word in sent]

            sentvec = self._mean(wordvecs)

            yield sentvec

    def _word2vec(self, word):
        try:
            word_vec = self.id2word[word]
            if len(word_vec) == self.WORD_DIM:
                return word_vec
            else:
                return np.zeros(self.WORD_DIM)

        except KeyError:
            return np.zeros(self.WORD_DIM)

    def _mean(self, wordvecs):
        try:
            sentvec = (reduce(add, wordvecs)) / len(wordvecs)

            return list(sentvec)

        except TypeError:
            # TODO 暫定処置
            return list(np.zeros(self.WORD_DIM))


class TfidfVectorizer(BaseEstimator, TransformerMixin):

    def fit(self):
        return self

    def transform(self, sents):
        """
        tf_idfによりsentssをベクトルに変換する
        :param corpus: list(list(str))
        :return:
        """
        # 変換するために、全ての単語のlistを生成
        words = sum(sents, [])
        word_collection = TextCollection(words)

        for sent in sents:
            yield {
                word: word_collection.tf_idf(word, sent)
                for word in sent
            }


class OneHotVectorizer(BaseEstimator, TransformerMixin):

    def fit(self):
        return self

    def transform(self, sents):
        for sent in sents:
            yield self._one_hot_vectorize(sent)

    def _one_hot_vectorize(self, sent):
        """
        one-hotによりsentenceをベクトルに変換する
        :param sent: list(str)
        :return:
        """

        return {
            token: True
            for token in sent
        }


class FreqVectorizer(BaseEstimator, TransformerMixin):

    def fit(self):
        return self

    def transform(self, sents):
        for sent in sents:
            yield self._freq_vectorize(sent)

    def _freq_vectorize(self, sent):
        """
        sentenceごとに単語の頻度分布をとることでベクトルに変換する
        :param sent: list(str)
        :return:
        """
        features = defaultdict(int)
        for token in sent:
            features[token] = +1

        return features
