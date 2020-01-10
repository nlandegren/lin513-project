"""This script processes raw text input to be used for training word
embeddings.

Usage Example:
    pre = Preprocessor(filename, window_size)
    skipgrams = pre.make_skipgrams(list_of_words)
    real_context = pre.positive_context(target_word_index, skipgrams)
    fake_context = pre.negative_context(target_word, window_size)
"""
import sys
import numpy as np
import math
from collections import Counter
import bz2


class Preprocessor(object):
    """A preprocessing unit that prepares training data for the word2vec
    skipgram model.

    Attributes:
        word_index: A dict object associating every unique word with a unique
        integer.
        index_word: Same as word_index but other way around.
        word_frequency: Keeps a tally of the occurrences of every unique word.
        window_size: The range in which to make skipgrams.
    """

    def __init__(self, data_file, window_size):
        """Initializes a Preprocessor object with a file name and window
        size."""
        self.word_index = {}
        self.index_word = {}
        self.word_frequency = Counter()
        self.window_size = window_size
        self.process_data(data_file)

    def __getitem__(self, word):
        """Gets the index assigned to a given word."""

        return self.word_index.get(word)

    def process_data(self, filenames):
        """Gives each unique word in a .txt file an index and saves
        it's frequency of occurrence.
        Args:
            filename: The name of a .txt file as a str object.
        """
        for filename in filenames:
            with open(filename, 'r') as fin:
                for line in fin:
                    line = line.split()
                    self.word_frequency.update(line)

        self.word_frequency = dict(self.word_frequency.most_common(200000))

        for k in self.word_frequency:
            self.word_index[k] = len(self.word_index)

        for k, v in self.word_index.items():
            self.index_word[v] = k
            self.word_frequency[v] = self.word_frequency.pop(k)

        print("data processed")

    def positive_context(self, word_index, skipgrams):
        """Retrieves relevant context for the target word.
        Args:
            word_index: The target word's position index in the line.
            skipgrams: A dict object where every word in the line of text is
            associated with a skipgram.
        Returns:
            The skipgram associated with the given word index.
        """
        real_context = skipgrams[word_index]

        return real_context

    def negative_context(self, target_word, prob_param = 0.75):
        """Generates random words as negative examples of context words.
        Args:
            target_word: A word represented by an int object.
            window_size
        """
        population = list(self.word_frequency.keys())

        # Removes the target word from the population.
        del population[target_word]

        # Computes the probability distribution for the words in the
        # vocabulary, minus the target word.
        freq_sum = sum([self.word_frequency[word]**prob_param for word in population])
        prob = [(float(self.word_frequency[word]**prob_param)/freq_sum) for word in population]

        # Sets number of negative samples per positive sample to be drawn,
        # default is 2.
        num_of_words = self.window_size - 1 * 2

        fake_context = np.random.choice(population, size = num_of_words, p = prob)

        return fake_context

    def make_skipgrams(self, line):
        """Generates all skip grams in a line given a window size."""

        # Equals the number of words on either side of the target word.
        w = int(self.window_size / 2 - 1) 

        skipgram_list = {}

        # for each word in the line, adds a 2-tuple to skipgram_list
        # containing the index of the word and a list of the word's
        # context words
        for i,word in enumerate(line):    
            # Gets the words in the window preceding the target word.
            pre_window = line[max(0, i-w):i]
            # Gets the words in the window following the target word.
            post_window = line[i: min(len(line), i+w+1)]
            del post_window[0] # Removes the target word.

            skipgram_list[i] =  pre_window+post_window

        return skipgram_list

    def subsample(self, word):
        """Calculates whether a sample word should be skipped in training.
        Args:
            word: A target word from the data, represented by an int object.
        Returns:
            False if the word should be discarded, True otherwise.
        """

        word_freq = self.word_frequency[word]/sum(self.word_frequency.values())
        threshold = 0.00001
        prob = 1 - math.sqrt(threshold/word_freq)

        if np.random.random() < prob:
            return False
        else:
            return True
