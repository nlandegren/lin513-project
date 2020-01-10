"""This script represents a classifier trainable with the word2vec skip gram
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
        target_matrix: A numpy.ndarray of 300 dimensions containing the
        embeddings that are updated for target word in the training data.
        context_matrix: A numpy.ndarray of 300 dimensions containing the
        embeddings that are updated for each context word in the training
        data.
    """

    def __init__(self, vocab_size):
        """Initiates a Classifier object with the size of the vocabulary."""

        self.target_matrix = np.random.rand(vocab_size, 300)
        self.context_matrix = np.random.rand(300, vocab_size)


    def train(self, target_word, real_context, fake_context): 
        """Uses received training data to train the embeddings of the
        classifier."""
        
        def estimate_prob(target_word, context_words):
            """Estimates the probability that the set of context words would appear in the
            context of word t.
            Args:
                t: A word represented by an int object.
                c: A word represented by an int object.
            Returns:
                The probability, as a float object, that context_words is a real
                context of the target word.
                """
            target_vector = self.target_matrix[target_word]
            context_prob = 1
            for word in context_words:
                context_vector = self.context_matrix[:,word]
                # Computes dot product between the target word's and context
                # word's vector.
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
            Args:a
                t: A word as an int object.
                c: A word as an int object.
                prob: the probability that context_words is a real context of
                the target word.
                label: 1 if c is a real context, 0 otherwise.
            """
            context_sum = sum([self.context_matrix[:,c] for c in context_words])

            # Controls how large each parameter adjustment is.
            step_param = 0.1
            
            target_vector = self.target_matrix[target_word]
            # Updates the target word vector.
            target_vector = target_vector - step_param*(prob - label)*context_sum
            for context_word in context_words:
                context_vector = self.context_matrix[:,context_word]
                # Updates the context word vector.
                context_vector = context_vector - step_param*(prob - label)*target_vector
            
        probability = estimate_prob(target_word, real_context)
        update_params(target_word, real_context, probability, 1)

        probability = estimate_prob(target_word, fake_context)
        update_params(target_word, fake_context, probability, 0)




