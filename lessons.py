"""Life lessons bigram model.
"""


import csv
from bigram_model import BigramModel
import random
import string


def main():
    corpus = load_data()

    bm = BigramModel()
    bm.fit(corpus)

    with open('fake_lessons.txt', 'w+') as out:
        for _ in range(1000):
            print(_)
            r1 = random.randint(10, 30)
            r2 = random.randint(2, 5)
            sentence = bm.gen_sentence(max_length=r1, start_word=None,
                                       num_top_words=r2)
            out.write(sentence + '\n')


def load_data():
    result = []
    with open('data.csv', 'rU') as fin:
        r = csv.reader(fin, delimiter=',')
        for line in r:
            if len(line) > 1:
                raise AttributeError('Data format is incorrect.')
            line = line[0].strip()
            line = line.translate(None, string.punctuation)
            line = line.lower()
            result.append(line)
    return result


if __name__ == '__main__':
    main()
