"""Outputs all l1 vectors with their translated to l2 vectors in parallel.

The module takes a word lexicon and two files containing word embeddings
and outputs two files that only contain parallel embeddings, i.e if a word
embedding has a correct translation according to the lexicon, the two word
embeddings are written out in parallel to their respective file. The format of
the contents of the lexicon file should be: l1_word l2_word\n. Outputs files
of the same name as the input vector files with added 'parallel' suffix.

    Usage:
    python3 make_parallel.py l1-l2_lexicon.txt l1.vec l2.vec
"""

import sys
from collections import defaultdict


def make_word_lexicon(filename):
    """Populates a dict object with translations from text file.

    Only one the first translation in the given file for a given word is saved.

    Args:
        filename: The pathway to a lexicon file as a str object.

    Returns:
        Two dict objects together making a two way lexicon.
    """
    l1_word_lexicon = {}
    l2_word_lexicon = {}
    with open(filename) as f:
        for line in f:
            line = line.split()
            # Saves only the first translation for a given word
            l1_word_lexicon.setdefault(line[0], line[1])
            l2_word_lexicon.setdefault(line[1], line[0])
    return l1_word_lexicon, l2_word_lexicon


def make_vec_lexicon(filename, word_lexicon):
    """Stores vectors in a dict if they have translations.
    Args:
        filename: The pathway to a file containing vectors.
        word_lexicon: A dict object containing word translations.
    Returns:
        A dict object with vectors which have translations in word_lexicon.
    """
    vec_lexicon = {}
    with open(filename) as f2:
        next(f2)
        for line in f2:
            line = line.split()
            # Populates word_lexicon with words as keys and their vectors as
            # values
            if line[0] in word_lexicon:
                vec_lexicon[line[0]] = line[1:]
    return vec_lexicon


def write_out(l1_word_lexicon, l1_vectors, l2_vectors, l1_fname, l2_fname):
    """Writes out vectors to files in parallel, if they have translations in
    their respective languages.
    Args:
        l1_word_lexicon: A dict object containing translation pairs.
        l1_vectors: A dict object containing words and their vectors.
        l2_vectors: A dict object containing words and their vectors.
        l1_fname: The filename of the l1 vector file.
        l2_fname: The filename of the l2 vector file.
    """
    num_of_vecs = 0
    with open('parallel'+l1_fname, 'w') as fout1:
        with open('parallel'+l2_fname, 'w') as fout2:
            # For a given translation pair, if we have the vectors for both
            # words, writes them out in parallel to their respective files
            for k, v in l1_word_lexicon.items():
                if k in l1_vectors and v in l2_vectors:
                    fout1.write(k + ' ' + ' '.join(l1_vectors[k])+'\n')
                    fout2.write(v + ' ' + ' '.join(l2_vectors[v])+'\n')


def main():
    """Matches vectors of correct translations and writes them to files.
    """
    l1_word_lexicon, l2_word_lexicon = make_word_lexicon(sys.argv[1])
    l1_vectors = make_vec_lexicon(sys.argv[2], l1_word_lexicon)
    l2_vectors = make_vec_lexicon(sys.argv[3], l2_word_lexicon)
    write_out(l1_word_lexicon, l1_vectors, l2_vectors,
              sys.argv[2], sys.argv[3])

    print('done')


if __name__ == '__main__':
    main()
