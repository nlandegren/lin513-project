"""A sanity test for your brand new word embeddings.

Takes a file of word embeddings and prints out the most similar words for a few
random word embeddings.
"""
import sys
import numpy as np
import random


def test_vectors():
    """Prints out 5 most similar vectors for 5 random vectors."""

    def distance(vec1, vec2):
        """Returns cosine distance between two vectors."""
        return np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

    vector_dict = {}
    with open(sys.argv[1]) as f:
        next(f)  # Skips the first line that contains meta info
        for line in f:
            line = line.strip('\n').split(' ')
            # Stores each word as key along with it's vector as value
            vector_dict[line[0]] = [float(i) for i in line[1:]]

    userword = input('press enter to test embeddings')
    # Loops until user types "n"
    while userword != 'n':
        # Picks five random vectors
        picks = random.choices(list(vector_dict), k=5)
        for word in picks:
            result = []
            for k, v in vector_dict.items():
                # Saves each word comparison as 3-tuple containing the two
                # words and their cosine distance, and appends to result list
                result.append((word, k, distance(vector_dict[word],
                               vector_dict[k])))
            # Prints out the 5 most similar words of the 5 randomized words
            for vec in reversed(sorted(result, key=lambda x: x[2])[-5:]):
                print(vec)
            print('----------------------------')
        userword = input('test again? y/n :')


test_vectors()
