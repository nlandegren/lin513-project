"""This script acts as a wrapper for the matrices used in the translation
model.

Usage example:
    vector_space1 = VectorSpace(list_of_vectors)
    vector_space2 = VectorSpace(list_of_vectors)
    vector_space1.align(vector_space2)
    vector_space1.distance(vector_space2)
"""

import numpy as np
from scipy.spatial import distance

class VectorSpace(object):
    """Represents a vector space for word embeddings.
    
    Attr:
        vocabulary: A list object of all unique words in the vector space.
        word_indexing: A dict object of words as str objects as keys with
        integers as values indexing their location in the matrix.
        matrix: A word embedding matrix as a numpy.ndarray.
        vec_positions: A list of int objects, used to keep track of the vectors
        positions in the matrix in case of rearrangement.
    """

    def __init__(self, vector_list):
        """Initiates a VectorSpace with a list containing lists, each of which
        contains a word as a str object its word embedding as a list
        object."""

        self.vocabulary = []
        self.word_indexing = {}
        self.matrix = np.zeros((len(vector_list), len(vector_list[0][1])))        
        self.vec_positions = list(range(len(vector_list)))
        # Populates the VectorSpace attributes with the word embeddings
        for i, vec in enumerate(vector_list):
            self.vocabulary.append(vec[0])
            self.word_indexing[vec[0]] = i
            self.matrix[i] = np.asarray(vec[1], dtype=float)


    def align(self, target_space):
        """Rotates the matrix of this instance of VectorSpace (self), towards
        the matrix of another VectorSpace object, and produces a mapping matrix,
        through singular value decomposition.
        Args:
            target_space: A VectorSpace object to estimate a mapping matrix
            with.
        Returns:
            A mapping matrix as a numpy.ndarray of shape (n,n) n being the
            number of dimensions of the word embeddings.
        """
        # computes dot product between transposed self and target space
        y = np.transpose(self.matrix).dot(target_space.matrix)
        # performs singular value decomposition on above computed dot product
        u, s, vh = np.linalg.svd(y, full_matrices=True)
        # produces mapping matrix
        mapping_matrix = u.dot(vh)

        return mapping_matrix

   
    def distance(self, target_space):
        """For each of the embeddings in the matrix of this VectorSpace object (self),
        finds it's nearest neighbor in a target space.
        
        Args:
            target_space: A VectorSpace object.

        Returns:
            For two matrices A and B, returns an A by B distance numpy.ndarray.
        """

        return distance.cdist(self.matrix, target_space.matrix, 'cosine')


