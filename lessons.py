"""Life lessons bigram model.
"""


import csv
import nltk
import random
import string

START = '<s>'
STOP = '</s>'


def main():
    corpus = clean(load_data())
    unigram_c = calc_unigrams(corpus)
    bigram_p = calc_bigram_p(corpus, unigram_c)
    gen_sentence(bigram_p, 10)


def gen_sentence(bigram_p, length):
    start_words = []
    for b, p in bigram_p.items():
        w1, w2 = b
        if w1 == START:
            start_words.append(w2)
    for w in start_words:
        sentence = [w]
        for i in range(length-1):
            w = get_next_word(bigram_p, w)
            sentence.append(w)
        sentence = [s for s in sentence if s and s != START and s != STOP]
        print(' '.join(sentence))


def get_next_word(bigram_p, w0, num_top_words=3):
    probs = []
    for b, p in bigram_p.items():
        w1, w2 = b
        if w1 == w0:
            probs.append((w2, p))
    if len(probs) == 0:
        return None
    top_words = sorted(probs, key=lambda x: x[1])[:num_top_words]
    r = random.randint(0, len(top_words)-1)
    return top_words[r][0]


def calc_unigrams(corpus):
    unigram_c = {}
    for sentence in corpus:
        for token in tokenize(sentence):
            if token not in unigram_c:
                unigram_c[token] = 0
            unigram_c[token] += 1
    unigram_c[START] = len(corpus)
    return unigram_c


def calc_bigram_p(corpus, unigram_c):
    temp = {}
    for sentence in corpus:
        tokens = tokenize(sentence)
        tokens = [START] + tokens + [STOP]
        bigrams = nltk.bigrams(tokens)
        for b in bigrams:
            if b not in temp:
                temp[b] = 0
            temp[b] += 1

    bigram_p = {}
    for token, count in temp.items():
        w1, w2 = token
        numer = count
        denom = unigram_c[w1]
        bigram_p[token] = numer / float(denom)

    validate_bigram_p(bigram_p, unigram_c)
    return bigram_p


def validate_bigram_p(bigram_p, unigram_c):
    vocab = set(unigram_c.keys())
    vocab.add(START)
    for v in vocab:
        s = 0
        for b, p in bigram_p.items():
            if b[0] == v:
                s += p
        assert abs(1 - s) < 1e-9


def clean(data):
    result = []
    for line in data:
        line = line.translate(None, string.punctuation)
        line = line.lower()
        result.append(line)
    return result


def tokenize(data):
    return data.split()


def load_data():
    result = []
    with open('data.csv', 'rU') as fin:
        r = csv.reader(fin, delimiter=',')
        for line in r:
            if len(line) > 1:
                raise AttributeError('Data format is incorrect.')
            result.append(line[0].strip())
    return result


if __name__ == '__main__':
    main()
