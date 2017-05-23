"""Bigram model:

https://en.wikipedia.org/wiki/Bigram
"""

import nltk
import random


class BigramModel(object):

    START = '<s>'
    STOP = '</s>'

# Public methods
# -----------------------------------------------------------------------------

    def __init__(self):
        pass

    def fit(self, corpus):
        self.unigram_c   = self._calc_unigrams(corpus)
        self.bigram_p    = self._calc_bigram_p(corpus)
        self.start_words = self._get_start_words()
        self.stop_words  = self._get_stop_words()

    def gen_sentence(self, max_length=10, start_word=None, num_top_words=3):
        if start_word:
            w = start_word
            sentence = [w]
        else:
            r = random.randint(0, len(self.start_words))
            w = self.start_words[r]
            sentence = [w]
        while len(sentence) < max_length:
            w = self._gen_next_word(self.bigram_p, w, num_top_words)
            if w is None:
                break
            sentence.append(w)
        sentence = [s for s in sentence if s != self.START and s != self.STOP]
        return ' '.join(sentence)

# Private methods
# -----------------------------------------------------------------------------

    def _gen_next_word(self, bigram_p, w0, num_top_words):
        probs = []
        for b, p in bigram_p.items():
            w1, w2 = b
            if w0 and w1 == w0:
                probs.append((w2, p))
            elif not w0 and w1 == self.START:
                probs.append((w2, p))
        if len(probs) == 0:
            return None
        top_words = sorted(probs, key=lambda x: x[1])[-num_top_words:]
        r = random.randint(0, len(top_words)-1)
        return top_words[r][0]

    def _calc_unigrams(self, corpus):
        unigram_c = {}
        for sentence in corpus:
            for token in _tokenize(sentence):
                if token not in unigram_c:
                    unigram_c[token] = 0
                unigram_c[token] += 1
        unigram_c[self.START] = len(corpus)
        return unigram_c

    def _calc_bigram_p(self, corpus):
        temp = {}
        for sentence in corpus:
            tokens = _tokenize(sentence)
            tokens = [self.START] + tokens + [self.STOP]
            bigrams = nltk.bigrams(tokens)
            for b in bigrams:
                if b not in temp:
                    temp[b] = 0
                temp[b] += 1

        bigram_p = {}
        for token, count in temp.items():
            w1, w2 = token
            numer = count
            denom = self.unigram_c[w1]
            bigram_p[token] = numer / float(denom)

        self._validate_bigram_p(bigram_p)
        return bigram_p

    def _get_start_words(self):
        words = []
        for b, p in self.bigram_p.items():
            w1, w2 = b
            if w1 == self.START:
                words.append(w2)
        return words

    def _get_stop_words(self):
        words = []
        for b, p in self.bigram_p.items():
            w1, w2 = b
            if w2 == self.STOP:
                words.append(w1)
        return words

    def _get_start_stop_words(self, symbol):
        words = []
        for b, p in self.bigram_p.items():
            w1, w2 = b
            if w1 == symbol:
                words.append(w2)
        return words

    def _validate_bigram_p(self, bigram_p):
        vocab = set(self.unigram_c.keys())
        vocab.add(self.START)
        for v in vocab:
            s = 0
            for b, p in bigram_p.items():
                if b[0] == v:
                    s += p
            assert abs(1 - s) < 1e-9
        self.vocab = vocab


def _tokenize(data):
    return data.split()
