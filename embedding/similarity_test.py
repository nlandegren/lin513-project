"""A similarity test script for words embeddings.

Used as a quick test to make sure your word embeddings aren't broken. Takes a
vector file and prints out the five vectors with least cosine distance to five
randomly picked vectors.

    Example:
    $ python3 similarity_test.py vector_file
"""

import sys
import numpy as np
import random


def test_vectors():
    """Prints out five most similar vectors for five random vectors.

    Reads in all the vectors from a given vector file, prints out similar
    vectors for a few random vectors, and prompts the user to test again.
    """
    vec_dict = get_vectors(sys.argv[1])
    main_loop(vec_dict)


def distance(vec1, vec2):
    """Returns cosine distance between two vectors.
    Args:
        vec1: A vector as a list of floats.
        vec2: A vector as a list of floats.
    Returns:
        The cosine distance between vec1 and vec2.
    """
    return np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))


def get_vectors(filename):
    """Opens a vector file and stores the vector in a dictionary.
    Args:
        filename: The name of a vector file.
    Returns:
        A dict object containing words as keys and vectors as values.
    """
    vector_dict = {}
    with open(filename) as f:
        next(f)
        for line in f:
            line = line.strip('\n').split()
            # Stores each word as key along with it's vector as value
            vector_dict[line[0]] = [float(i) for i in line[1:]]
    return vector_dict


def main_loop(vector_dict):
    """A loop performing the similarity tests until user enters 'n'.
    Args:
        vector_dict: A dict object containing words associated with their
        vectors.
    """
    userword = input('press enter to test embeddings')
    # Loops until user enters 'n'
    while userword != 'n':
        # Picks five random vectors
        picks = [random.choice(list(vector_dict)) for i in range(5)]
        for word in picks:
            result = []
            for k, v in vector_dict.items():
                # Saves each word comparison as 3-tuple containing the two
                # words and their cosine distance, and appends to result
                # list
                result.append((word, k, distance(vector_dict[word],
                               vector_dict[k])))
            # Prints out the 5 most similar words of the 5 randomized words
            for vec in reversed(sorted(result, key=lambda x: x[2])[-5:]):
                print(vec)
            print('----------------------------')
        userword = input('test again? y/n :')


if __name__ == '__main__':
    test_vectors()
