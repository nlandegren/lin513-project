"""Trains and evaluates a translation model based on lexicon induction.

The script takes VectorSpace objects as input and aims to find an optimal
mapping matrix between them.

Usage example:
    translator = Translator(seed_data)
    translator.train(l1_vectorspace, l2_vectorspace)
    translator.evaluate(l1_vectorspace, l2_vectorspace)
"""

import numpy as np
from scipy.spatial import distance
import time
from collections import Counter

class Translator(object):
    """Represents a translation model based on iterative lexicon induction.

    Attr:
        mapping_matrix: A mapping matrix as a numpy.ndarray of shape (n,n),
        n being the number of dimensions of the word embeddings.
    """

    def __init__(self, seed_data, train_iters = 10):
        """Initiates a Translator object with a list of two parallell VectorSpace
        objects used as a 'seed lexicon' for producing the first mapping
        matrix."""

        self.mapping_matrix = seed_data[0].align(seed_data[1])
        self.train_iters = train_iters

    def train(self, source_vs, target_vs):
        """Iteratively produces a better and better mapping matrix, replacing
        the attribute mapping_matrix each time.
        Args:
            source_vs: A VectorSpace object to align towards the target space.
            target_vs: A VectorSpace object for alignment towards.
        """
        def nearest_neighbor(source_matrix, target_matrix):
            
            distance_matrix = 1 - distance.cdist(source_matrix, target_matrix, 'cosine')
            sorted_dist_matrix = np.sort(distance_matrix, axis=1)
            rt = np.mean(sorted_dist_matrix[:,-10:], axis=1, keepdims=True)
            sorted_dist_matrix = np.sort(distance_matrix, axis=0)
            st = np.mean(sorted_dist_matrix[-10:], axis=0, keepdims=True)
            distance_matrix = distance_matrix - rt - st
            
            return np.argmax(distance_matrix, axis=1)

        for iterations in range(self.train_iters):    

            # Uses the mapping matrix of the Translator object to estimate a
            # rotation from the source vector space towards the target vector
            # space.
            x = source_vs.matrix.dot(self.mapping_matrix)

            translation = nearest_neighbor(x, target_vs.matrix)
            
            # Rearranges the rows of the target matrix so that each row vector
            # in the source matrix is parallell with its nearest neighbor in
            # the target matrix.
            target_vs.matrix = target_vs.matrix[translation]
            
            # Keeps track of the positions of the rearranged word embeddings.
            target_vs.vec_positions = [target_vs.vec_positions[i] for i in translation]
            
            # Produces a better mapping matrix from the now (hopefully) more
            # parallell source and target spaces.
            self.mapping_matrix = source_vs.align(target_vs) 
            
            self.print_correctness(target_vs.vec_positions)


    def evaluate(self, source_vs, target_vs):
        """Evaluates the mapping matrix produced in training of the Translator
        object and prints a score of correct translations.
        Args:
            source_vs: A VectorSpace object to align towards the target space. 
            target_vs: A VectorSpace object for alignment towards.
        """
        
        # Rotates the source matrix towards the target matrix with the trained
        # mapping matrix.
        source_vs.matrix = source_vs.matrix.dot(self.mapping_matrix)
        
        # For each of the embeddings in the rotated source space, finds it's
        # nearest neighbor in the target space.
        translation = np.argmin(source_vs.distance(target_vs), axis=0)
        # Rearranges the rows of the target matrix so that each row vector
        # in the source matrix is parallell with its nearest neighbor in
        # the target matrix.
        target_vs.matrix = target_vs.matrix[translation]

        # Keeps track of the positions of the rearranged word embeddings.
        target_vs.vec_positions = [target_vs.vec_positions[i] for i in translation]

        self.print_correctness(target_vs.vec_positions)
    
    def print_correctness(self, target_matrix_positions):
        rearranged_positions = target_matrix_positions
        correct_positions = range(len(target_matrix_positions))

        c = 0
        for i, j in zip(correct_positions, rearranged_positions):
            if i == j:
                c += 1
        print("correctness: ", c, '/', len(correct_positions))
       

