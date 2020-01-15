"""Aligns word vectors.

A module that takes two files containing word embeddings and a word lexicon
and outputs two files that only contain aligned embeddings, i.e if a word
embedding has a correct translation according to the lexicon, the two word
embeddings are written out in parallell to their respective file.

    Usage:
    python3 make_parallel.py l1-l2_lexicon.txt l1.vec l2.vec

"""

import sys
from collections import defaultdict


def make_word_lexicon(filename, l1_word_lexicon, l2_word_lexicon):
    """Populates l1_word_lexicon and l2_word_lexicon with their l1 words and
    respective l2 translations and vice versa."""
    with open(filename) as f1:
        for line in f1:
            line = line.split()
            l1_word_lexicon.setdefault(line[0], line[1])
            l2_word_lexicon.setdefault(line[1], line[0])


def make_vec_lexicon(filename, word_lexicon, vec_lexicon):
    """If we have a translation for the l1-vector, adds the vector to
    l1_vectors."""
    with open(filename) as f2:
        next(f2)
        for line in f2:
            line = line.split()
            if line[0] in word_lexicon:
                vec_lexicon[line[0]] = line[1:]


def l2_vec_lexicon():
    """If there is a translation for the l2-vector, adds the vector to
    l2_vectors."""

    with open(sys.argv[3]) as f3:
        next(f3)
        for line in f3:
            line = line.split()
            if line[0] in l2_word_lexicon:
                l2_vectors[line[0]] = line[1:]


def write_out(l1_word_lexicon, l1_vectors, l2_vectors):
    """If we have both of the vectors for an l1-l2 pair, adds them in
    parallell to each of l1_vec and l2_vec."""
    num_of_vecs = 0
    with open('l1.vec', 'w') as fout1:
        with open('l2.vec', 'w') as fout2:
            for k, v in l1_word_lexicon.items():
                if k in l1_vectors and v in l2_vectors:
                    num_of_vecs += 1
                    fout1.write(k + ' ' + ' '.join(l1_vectors[k])+'\n')
                    fout2.write(v + ' ' + ' '.join(l2_vectors[v])+'\n')


def main():

    l1_word_lexicon = {}
    l2_word_lexicon = {}

    l1_vectors = {}
    l2_vectors = {}

    make_word_lexicon(sys.argv[1], l1_word_lexicon, l2_word_lexicon)
    make_vec_lexicon(sys.argv[2], l1_word_lexicon, l1_vectors)
    make_vec_lexicon(sys.argv[3], l2_word_lexicon, l2_vectors)
    write_out(l1_word_lexicon, l1_vectors, l2_vectors)

    print('done')


if __name__ == '__main__':
    main()
