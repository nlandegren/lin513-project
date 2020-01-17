"""Contains the VectorSpace class.

Objects of the VectorSpace class should be used to represent a set of
vectors. Vectors are stored in a numpy ndarray.

    Usage:

    vector_space1 = VectorSpace(list_of_vectors)
    vector_space2 = VectorSpace(list_of_vectors)
    vector_space1.align(vector_space2)
"""

import numpy as np


class VectorSpace(object):
    """Represents a vector space for word embeddings.

    Attributes:
        vocabulary: A list containing all unique words as strings.
        word_indexing: A dict containing every word in the vocabulary, each
        assigned an int as an index.
        matrix: A numpy ndarray containing word embeddings as row vectors.
        vec_positions: A list of integers, representing the positions of the
        vectors in the matrix.
    """

    def __init__(self, vector_list, dim=300):
        """Initiates a VectorSpace object with a list of vectors."""

        self.vocabulary = []
        self.word_indexing = {}
        self.matrix = np.zeros((len(vector_list), dim))
        self.vec_positions = list(range(len(vector_list)))

        # Populates the VectorSpace attributes with the word embeddings
        for i, vec in enumerate(vector_list):
            self.vocabulary.append(vec[0])
            self.word_indexing[vec[0]] = i
            self.matrix[i] = np.asarray(vec[1:], dtype=float)

    def align(self, target_space):
        """Estimates a mapping matrix between two vector spaces.

        Produces mapping matrix between this VectorSpace object (self),
        and another.

        Args:
            target_space: A VectorSpace to estimate mapping matrix with.

        Returns:
            A mapping matrix as a numpy.ndarray.
        """
        # computes dot product between this object's matrix and the target's
        dot_product = np.transpose(self.matrix).dot(target_space.matrix)
        # performs singular value decomposition on above computed dot product
        u, s, vh = np.linalg.svd(dot_product, full_matrices=True)
        # returns mapping matrix
        return u.dot(vh)
