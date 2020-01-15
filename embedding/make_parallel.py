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


def make_word_lexicon(filename):
    """Populates l1_word_lexicon and l2_word_lexicon with their l1 words and
    respective l2 translations and vice versa."""
    word_lexicon = {}
    with open(filename) as f:
        for line in f:
            line = line.split()
            word_lexicon.setdefault(line[0], line[1])
    return word_lexicon


def make_vec_lexicon(filename, word_lexicon):
    """If we have a translation for the l1-vector, adds the vector to
    l1_vectors."""
    vec_lexicon = {}
    with open(filename) as f2:
        next(f2)
        for line in f2:
            line = line.split()
            if line[0] in word_lexicon:
                vec_lexicon[line[0]] = line[1:]
    return vec_lexicon


def write_out(l1_word_lexicon, l1_vectors, l2_vectors):
    """If we have both of the vectors for an l1-l2 pair, adds them in
    parallell to each of l1_vec and l2_vec."""
    num_of_vecs = 0
    with open('l1.vec', 'w') as fout1:
        with open('l2.vec', 'w') as fout2:
            for k, v in l1_word_lexicon.items():
                if k in l1_vectors and v in l2_vectors:
                    print(k, v)
                    fout1.write(k + ' ' + ' '.join(l1_vectors[k])+'\n')
                    fout2.write(v + ' ' + ' '.join(l2_vectors[v])+'\n')


def main():
    l1_word_lexicon = make_word_lexicon(sys.argv[1])
    l2_word_lexicon = make_word_lexicon(sys.argv[2])
    l1_vectors = make_vec_lexicon(sys.argv[2], l1_word_lexicon)
    l2_vectors = make_vec_lexicon(sys.argv[3], l2_word_lexicon)
    write_out(l1_word_lexicon, l1_vectors, l2_vectors)

    print('done')


if __name__ == '__main__':
    main()
