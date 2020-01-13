"""This module represents a classifier trainable with the word2vec skipgram
algorithm.

Usage Example:
    classifier = Classifier(vocabulary_size)
    classifier.train(target_word, real_context, fake_context)
"""
import numpy as np
from math import exp, log


class Classifier(object):
    """Represents a trainable classifier.

    Attributes:
        target_matrix: A numpy.ndarray containing word embeddings as row
        vectors.
        context_matrix: A numpy.ndarray containing word embeddings as column
        vectors.
    """

    def __init__(self, vocab_size, dim=300):
        """Initiates a Classifier object with vocabulary and dimension size."""

        self.target_matrix = np.random.rand(vocab_size, dim)
        self.context_matrix = np.random.rand(dim, vocab_size)

    def train(self, target_word, real_context, fake_context): 
        """Uses received training data to train the embeddings of the
        classifier.
        Args:
            target_word: An int object representing a target word training
            example.
            real_context: A skipgram as a list of int objects representing
            actual context words of the target word.
            fake_context: A list of int objects randomly picked from the
            vocabulary, serves as negative training example.
        """

        def estimate_prob(target_word, context_words):
            """Estimates the probability that context_words is an actual
            skipgram of the target word.
            Args:
                target_word: A word represented by an int object.
                context_words: A list of words represented by int objects.
            Returns:
                The probability, as a float object, that context_words is a
                real context of the target word."""
            target_vector = self.target_matrix[target_word]
            context_prob = 1
            for word in context_words:
                context_vector = self.context_matrix[:, word]
                # Computes dot product between the vectors of target and
                # context word
                dot_product = target_vector.dot(context_vector)
                # Turns the dot product into a probability with the sigmoid
                # function.
                word_prob = 1 / (1 + exp(-dot_product))
                # Keeps a tally of the total probability of the set of context
                # words.
                context_prob = context_prob * word_prob

            return context_prob

        def update_params(target_word, context_words, prob, label):
            """Updates the word embeddings of the target and context matrices.
            Args:
                target_word: A word represented by an int object.
                context_words: A list of words represented by int objects.
                prob: The probability that context_words is a real context of
                the target word.
                label: 1 if context_words is a real context, 0 otherwise.
            """
            context_sum = sum([self.context_matrix[:, c] for c
                              in context_words])
            # Controls how large each parameter adjustment is.
            step_param = 0.1
            target_vector = self.target_matrix[target_word]
            # Updates the target word vector.
            target_vector = (target_vector - step_param*(prob - label) *
                             context_sum)
            for context_word in context_words:
                context_vector = self.context_matrix[:, context_word]
                # Updates the context word vector.
                context_vector = (context_vector - step_param*(prob - label) *
                                  target_vector)
        # Gets the probability of the real context
        probability = estimate_prob(target_word, real_context)
        # Updates parameters of the target word and real context
        update_params(target_word, real_context, probability, 1)
        # Updates parameters of the target word and fake context
        update_params(target_word, fake_context, 1-probability, 0)
