"""Trains and evaluates a translation model based on aligning word embeddings.

Translator objects takes VectorSpace objects as input and aims to find an
optimal mapping matrix between them using singular value decomposition.

    Usage example:
    translator = Translator(seed_data)
    translator.train(l1_vectorspace, l2_vectorspace)
    translator.evaluate(l1_vectorspace, l2_vectorspace)
"""

import numpy as np
from scipy.spatial import distance


class Translator(object):
    """Represents a translation model based on vector space alignment.

    Attributes:
        mapping_matrix: A mapping matrix as a numpy.ndarray.
        train_iters: Number of iterations to be ran in training.
    """

    def __init__(self, seed_data, train_iters=30):
        """Initializes a Translator object with an initial set of training
        examples."""

        self.mapping_matrix = seed_data[0].align(seed_data[1])
        self.train_iters = train_iters

    def train(self, source_vs, target_vs):
        """Learns a mapping matrix between two vector spaces.
        Args:
            source_vs: A VectorSpace object to align.
            target_vs: A VectorSpace object to be aligned with.
        """
        for iterations in range(self.train_iters):

            # Uses the mapping matrix to estimate a rotation of the source
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

            print(self.print_correctness(target_vs.vec_positions))

    def evaluate(self, source_vs, target_vs):
        """Evaluates the trained mapping matrix on test data.

        Args:
            source_vs: A VectorSpace object containing test data.
            target_vs: A VectorSpace object containing test data.
        """

        # Uses the mapping matrix to estimate a rotation of the source
        # space towards the target space
        source_vs.matrix = source_vs.matrix.dot(self.mapping_matrix)

        # Gets nearest neighbor for each vector in the source space
        translation = self.nearest_neighbor(source_vs.matrix, target_vs.matrix)

        # Rearranges the target space matrix to be parallell with the
        # source space
        target_vs.matrix = target_vs.matrix[translation]

        # Keeps track of the positions of the rearranged word embeddings
        target_vs.vec_positions = [target_vs.vec_positions[i] for i in
                                   translation]

        print(self.print_correctness(target_vs.vec_positions))

    def nearest_neighbor(self, source_matrix, target_matrix):
        """Finds nearest neighbor in the target matrix.

        Uses cosine similarity together with a correcting score to account
        for semantic hubs in the vector space.
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

        # Adjusts similarity score with average score
        similarity_matrix = similarity_matrix - rt - st

        return np.argmax(similarity_matrix, axis=1)

    def print_correctness(self, matrix_positions):
        """Counts and returns number of correct translations."""
        correct_pos = range(len(matrix_positions))
        correct_translations = sum([1 for i, j in zip(correct_pos,
                                    matrix_positions) if i == j])
        return 'correctness = '+str(correct_translations / len(
                matrix_positions))
