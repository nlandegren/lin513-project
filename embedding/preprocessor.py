"""Contains the class Preprocessor.

Objects of the class are used for processing raw text input to be used for
training word embeddings. Words are converted to vocabulary indexes
represented by ints and training data is prepared from each line of text fed
to the methods of the class, in particular the skipgram method. Each method
taking words as input expect them to be represented by int objects.

A Preprocessor object is initialized with the entirety of the data and passes
through all of it once before any training data can be retrieved.

    Example:
    pre = Preprocessor(filename, window_size=10)
    skipgrams = pre.make_skipgrams(list_of_words_as_ints)
    real_context = pre.positive_context(target_word_position_in_line,
                                        skipgrams)
    fake_context = pre.negative_context(target_word)
"""

import numpy as np
import math
from collections import Counter


class Preprocessor(object):
    """Represents a preprocessing unit for preparing training data.

    Attributes:
        word_index: A dict object associating every unique word with a unique
        int object.
        index_word: Same as word_index but reversed keys and values.
        word_frequency: A Counter object containing each word's vocabulary
        index and it's frequency of occurrence.
        window_size: The size of each skipgram window as an int object,
        default is 10.
    """

    def __init__(self, data_file, window_size=10):
        """Initializes a Preprocessor object with a file name and window size.
        """
        self.word_index = {}
        self.index_word = {}
        self.word_frequency = Counter()
        self.window_size = window_size
        self.process_data(data_file)

    def __getitem__(self, word):
        """If word is in 200k most common, returns the words index, else None.
        """
        return self.word_index.get(word)

    def process_data(self, filenames):
        """Populates the objects attributes with data from text file.

        Reads through all the data and saves a unique index and the
        word frequencies of the 200k most common words.

        Args:
            filenames: A list object of pathways to text files as str objects.
        """
        # Saves word frequencies
        for filename in filenames:
            with open(filename, 'r') as fin:
                for line in fin:
                    line = line.lower().split()
                    self.word_frequency.update(line)
        # Saves only the 200k most common words
        self.word_frequency = dict(self.word_frequency.most_common(200000))
        # Gives each word an index
        for k in self.word_frequency:
            self.word_index[k] = len(self.word_index)

        for k, v in self.word_index.items():
            # Makes reversed index dict for translation back to words
            self.index_word[v] = k
            # Replaces each word (key) in word_frequency with its assigned
            # index
            self.word_frequency[v] = self.word_frequency.pop(k)

        print("data processed")

    def positive_context(self, index, skipgrams):
        """Retrieves the skipgram for a given word in a line of text.
        Args:
            index: The target word's position in the line of text the
            skipgrams were retrieved from, as an int object.
            skipgrams: A dict object where every word index in the line of
            text is associated with a skipgram.
        Returns:
            The skipgram associated with the given word index.
        """
        return skipgrams[index]

    def negative_context(self, target_word, prob_param=0.75):
        """Picks random words as negative examples of context words.
        Args:
            target_word: A word represented by an int object.
            prob_param: A parameter adjusting the probability of picking a
            word from the vocabulary.
        """
        # The words in the vocabulary to be randomly picked from
        population = list(self.word_frequency.keys())
        # Excludes the target word itself from the population
        del population[target_word]

        # Computes the propability with which to pick from the population
        freq_sum = sum([self.word_frequency[word]**prob_param for word in
                        population])
        prob = [(float(self.word_frequency[word]**prob_param)/freq_sum) for
                word in population]

        # Sets number of negative samples per positive sample to be drawn,
        # default is 2
        num_of_words = self.window_size - 1 * 2

        return np.random.choice(population, size=num_of_words,
                                p=prob)

    def make_skipgrams(self, line):
        """Generates all skip grams in a line given a window size.
        Args:
            line: A list of words represented by their index int objects.
        Returns:
            A dict object containing every word in the line of text associated
            with it's skipgram.
        """

        # Equals the number of words on either side of the target word.
        w = int(self.window_size / 2 - 1)

        skipgrams = {}

        # Populates the skipgram dict with the line position index of a word
        # as key and it's skipgram as value
        for i, word in enumerate(line):
            # Gets the words in the window preceding the target word
            pre_window = line[max(0, i-w):i]
            # Gets the words in the window following the target word
            post_window = line[i: min(len(line), i+w+1)]
            del post_window[0]  # Removes the target word

            skipgrams[i] = pre_window+post_window

        return skipgrams

    def subsample(self, word):
        """Calculates whether a sample word should be skipped in training.

        Based on a probability computed from their frequency of occurrence,
        more common words will be skipped more frequently.

        Args:
            word: A target word from the data, represented by it's index int
            object.

        Returns:
            False if the word should be discarded, True otherwise.
        """
        word_freq = self.word_frequency[word]/sum(self.word_frequency.values())
        threshold = 0.00001  # Magic number described in the word2vec paper
        prob = 1 - math.sqrt(threshold/word_freq)

        return np.random.random() > prob
