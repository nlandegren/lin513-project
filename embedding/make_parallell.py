"""Aligns word vectors.

A module that takes two files containing word embeddings and a word lexicon
and outputs two files that only contain aligned embeddings, i.e if a word
embedding has a correct translation according to the lexicon, the two word
embeddings are written out in parallell to their respective file.
"""

import sys
from collections import defaultdict

l1_word_lexicon = {}
l2_word_lexicon = {}

l1_vectors = {}
l2_vectors = {}

l1_vec = []
l2_vec = []


def make_word_lexicon():
    """Populates l1_word_lexicon and l2_word_lexicon with their l1 words and
    respective l2 translations."""
    with open(sys.argv[1]) as f1:
        for line in f1:
            line = line.split()
            word_lexicon.setdefault(line[0], line[1])
            l2_word_lexicon.setdefault(line[1], line[0])


def l1_vec_lexicon():
    """If we have a translation for the l1-vector, adds the vector to
    l1_vectors."""
    with open(sys.argv[2]) as f2:
        next(f2)
        for line in f2:
            line = line.split()
            if line[0] in word_lexicon:
                l1_vectors[line[0]] = line[1:]
            else:
                pass


def l2_vec_lexicon():
    """If there is a translation for the l2-vector, adds the vector to
    l2_vectors."""

    with open(sys.argv[3]) as f3:
        next(f3)
        for line in f3:
            line = line.split()
            if line[0] in l2_word_lexicon:
                l2_vectors[line[0]] = line[1:]
            else:
                pass


def match_vectors():
    """If we have both of the vectors for an l1-l2 pair, adds them in
    parallell to each of l1_vec and l2_vec."""
    for k, v in word_lexicon.items():
        if k in l1_vectors and v in l2_vectors:
            l1_vec.append([k, l1_vectors[k]])
            l2_vec.append([v, l2_vectors[v]])


def main():
    make_word_lexicon()
    kl1_vec_lexicon()
    l2_vec_lexicon()
    match_vectors()

    # Writes each of the lists to their respective text files
    with open('l1.vec', 'wb') as f:
        for vec in l1_vec:
            f.write(''.join(vec))

    with open('l2.vec', 'wb') as f:
        for vec in l2_vec:
            f.write(''.join(vec))

    print('length of l1_vec', len(l1_vec))
    print('length of l2_vec', len(l2_vec))
    print('done')


if __name__ == '__main__':
    main()
