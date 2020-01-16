"""Trains and evaluates a translation model based on aligning word embeddings.

Translator objects takes two VectorSpace objects as input and aims to find an
optimal mapping matrix between them using singular value decomposition.

    Usage example:
    translator = Translator(seed_data)
    translator.train(l1_vectorspace, l2_vectorspace)
    translator.test(l1_vectorspace, l2_vectorspace)
"""

import numpy as np
from scipy.spatial import distance


class Translator(object):
    """Represents a translation model based on vector space alignment.

    Attributes:
        mapping_matrix: A mapping matrix as a numpy.ndarray.
        train_iters: Number of iterations to be ran in training.
    """

    def __init__(self, seed_data):
        """Initializes a Translator object with an initial set of training
        examples."""

        self.mapping_matrix = seed_data[0].align(seed_data[1])

    def train(self, source_vs, target_vs, train_iters=10):
        """Iteratively produces a better mapping matrix.
        Args:
            source_vs: A VectorSpace object containing unaligned vectors for
            training.
            target_vs: A VectorSpace object containing unaligned vectors for
            training.
            train_iters: Number of iterations to perform in training.
        """
        for iterations in range(train_iters):
            self.estimate_translation(source_vs, target_vs)
            print("correctness in training:",
                  self.correctness(target_vs.vec_positions))
        print("mapping learned")

    def test(self, source_vs, target_vs):
        """Evaluates the trained mapping matrix on test data.
        Args:
            source_vs: A VectorSpace object containing unaligned vectors for
            testing.
            target_vs: A VectorSpace object containing unaligned vectors for
            testing.
        """
        self.estimate_translation(source_vs, target_vs)
        print("correctness in testing:",
              self.correctness(target_vs.vec_positions))

    def estimate_translation(self, source_vs, target_vs):
        """Learns a new mapping matrix between two vector spaces.
        Args:
            source_vs: A VectorSpace object containing vectors for training/
            testing.
            target_vs: A VectorSpace object containing vectors for training/
            testing.
        """
        # Uses an initial mapping matrix to estimate a rotation of the source
        # space towards the target space
        x = source_vs.matrix.dot(self.mapping_matrix)

        # Gets nearest neighbor for each vector in the source space
        translation = self.nearest_neighbor(x, target_vs.matrix)
        # Rearranges the target space matrix to be parallell with the
        # source space
        target_vs.matrix = target_vs.matrix[translation]

        # Keeps track of the positions of the rearranged word embeddings
        target_vs.vec_positions = [target_vs.vec_positions[i] for i in
                                   translation]
        # Computes a better mapping matrix with the aligned spaces
        self.mapping_matrix = source_vs.align(target_vs)

    def nearest_neighbor(self, source_matrix, target_matrix):
        """Finds nearest neighbor in the target matrix.

        Uses cosine similarity together with a correcting score to account
        for semantic hubs in the vector space.
        Args:
            source_matrix: A numpy ndarray containing row vectors of a newly
            rotated vector space.
            target_matrix: A numpy ndarray representing the vector space in
            which to find nearest neighbors.
        Returns:
            A semantic hub corrected similarity matrix containing for each
            vector in the source matrix the index of the most similar target
            vector.
        """
        # computes cosine similarity between each vector
        similarity_matrix = 1 - distance.cdist(source_matrix,
                                               target_matrix, 'cosine')
        # Computes average similarity between each vector and their
        # ten most similar vectors for each of the vector spaces
        sorted_dist_matrix = np.sort(similarity_matrix, axis=1)
        rt = np.mean(sorted_dist_matrix[:, -10:], axis=1, keepdims=True)
        sorted_dist_matrix = np.sort(similarity_matrix, axis=0)
        st = np.mean(sorted_dist_matrix[-10:], axis=0, keepdims=True)

        # Adjusts similarity score with average similarity
        similarity_matrix = similarity_matrix - rt - st

        return np.argmax(similarity_matrix, axis=1)

    def correctness(self, matrix_positions):
        """Counts and returns number of correct translations.
        Args:
            matrix_positions: A list of int objects representing the position
            of each vector in it's matrix.
        Returns:
            A str object stating the correctness of the translations as a
            decimal number ratio.
        """
        correct_pos = range(len(matrix_positions))
        correct_translations = sum([1 for i, j in zip(correct_pos,
                                    matrix_positions) if i == j])
        return str(correct_translations / len(matrix_positions))
