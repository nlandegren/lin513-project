"""The main module of the translator program.

The module acts as a user interface for the training and testing of a
Translator object.

Usage: python3 main.py l1_vector_file l2_vector_file
"""
import numpy as np
from vectorspace import VectorSpace
from translator import Translator
import sys


def main():
    l1_vectors = get_data(sys.argv[1])
    l2_vectors = get_data(sys.argv[2])
    l1_train_vs, l1_test_vs, l1_seed_vs = divide_data(l1_vectors)
    l2_train_vs, l2_test_vs, l2_seed_vs = divide_data(l2_vectors)
    # Prints data size for training, test and seed sets
    print(("Using training set of {} words, test set of {} words, "
           "seed set of {} words".format(len(l1_train_vs.matrix),
                                         len(l1_test_vs.matrix),
                                         len(l1_seed_vs.matrix))))

    translator = Translator(l1_seed_vs, l2_seed_vs)
    translator.train(l1_train_vs, l2_train_vs, train_iters=5)
    translator.test(l1_test_vs, l2_test_vs)


def get_data(filename):
    """Returns a list of all the lines in a file.
    Args:
        filename: The name of a text file containing vectors.
    Returns:
        A list containing each line (vector) as a list object.
    """
    with open(filename, 'r') as f:
        next(f)
        return [vec.strip().split() for vec in f]


def divide_data(vector_list):
    """Divides the data into training, test and seed data.
    Args:
        vector_list: A list of vectors as list objects.
    Returns:
        A tuple of the vector list split up into three parts, one each for
        training, testing and seed data, default is 40%/40%/20% split.
    """
    total_data = len(vector_list)

    training_slice = int(0.4*total_data)
    test_slice = int(0.4*total_data)
    seed_slice = int(0.2*total_data)

    # Initiates three VectorSpace objects with their own slice of the
    # vector_list training data
    training_data = VectorSpace(vector_list[:training_slice])
    test_data = VectorSpace(vector_list[training_slice:training_slice
                            + test_slice])
    seed_data = VectorSpace(vector_list[training_slice+test_slice:
                            training_slice+test_slice+seed_slice])

    return training_data, test_data, seed_data


if __name__ == '__main__':
    main()
