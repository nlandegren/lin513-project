"""Trains and evaluates a translation model based on aligning word embeddings.

Translator objects takes two VectorSpace objects as input and aims to find an
optimal mapping matrix between them using singular value decomposition.

    Example:
    translator = Translator(l1_vectorspace, l2_vectorspace)
    translator.train(l1_vectorspace, l2_vectorspace)
    translator.test(l1_vectorspace, l2_vectorspace)

Each vectorspace argument in the above example is expected to be a VectorSpace
object."""

import numpy as np
from scipy.spatial import distance


class Translator(object):
    """Represents a trainable translator based on vector space alignment.

    Attributes:
        mapping_matrix: A mapping matrix as a numpy.ndarray.
    """

    def __init__(self, source_vs, target_vs):
        """Initializes a Translator object with an initial set of training
        examples."""

        self.mapping_matrix = source_vs.align(target_vs)

    def train(self, source_vs, target_vs, train_iters=10):
        """Iteratively produces a better mapping matrix.
        Args:
            source_vs, target_vs: A VectorSpace object containing unaligned
            vectors for training.
            train_iters: An int specifying the number of iterations to perform
            in training.
        """
        for iterations in range(train_iters):
            # Trains the attribute mapping_matrix
            self.estimate_translation(source_vs, target_vs)
            print("correctness in training:",
                  self.correctness(target_vs.vec_positions))
        print("mapping learned")

    def test(self, source_vs, target_vs):
        """Evaluates the trained mapping matrix on test data.
        Args:
            source_vs, target_vs: A VectorSpace object containing unaligned
            vectors for testing.
        """
        self.estimate_translation(source_vs, target_vs)
        print("correctness in testing:",
              self.correctness(target_vs.vec_positions))

    def estimate_translation(self, source_vs, target_vs):
        """Learns a new mapping matrix between two vector spaces.
        Args:
            source_vs, target_vs: A VectorSpace object containing vectors for
            training/testing.
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
            An array of the indices of the most similar vector in the target
            space for each vector in the source space.
        """
        # computes cosine similarity between each vector
        similarity_matrix = 1 - distance.cdist(source_matrix,
                                               target_matrix, 'cosine')
        # Computes average similarity between each vector and their
        # ten most similar vectors for the source and target vector spaces
        # respectively
        sorted_similarity_matrix = np.sort(similarity_matrix, axis=1)
        source_top_ten_mean = np.mean(sorted_similarity_matrix[:, -10:],
                                      axis=1, keepdims=True)

        sorted_similarity_matrix = np.sort(similarity_matrix, axis=0)
        target_top_ten_mean = np.mean(sorted_similarity_matrix[-10:],
                                      axis=0, keepdims=True)

        # Adjusts similarity score with average similarity
        similarity_matrix = similarity_matrix - target_top_ten_mean
        - source_top_ten_mean

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
