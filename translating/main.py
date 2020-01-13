"""The main module acting as a user interface for training and evaluating a
Translator object.

Usage: python3 main.py l1_vector_file l2_vector_file
"""
import numpy as np
from vectorspace import VectorSpace
from translator import Translator
import sys

# Loads vectors
l1_vectors = []
l2_vectors = []

with open(sys.argv[1], 'rb') as f1:
    for vec in f1:
        l1_vectors.append(vec)

with open(sys.argv[2], 'rb') as f2:
    for vec in f1:
        l2_vectors.append(vec)

# sets the total amount of words to use (for each vector space)
total_data = 5000

# sets in what proportions the data will be split up, default is training 40%,
# testing 40%, seed dictionary 100 words
training_slice = int(0.5*total_data)
test_slice = int(0.5*total_data)
seed_slice = 25

print("Using seed dictionary of {} words.".format(seed_slice))
print("Using training set of {} words.".format(training_slice))
print("Using test set of {} words.".format(test_slice))

# divides the data according to the slice parameters set above
l1_training_data = l1_vectors[:training_slice]
l2_training_data = l2_vectors[:training_slice]

l1_test_data = l1_vectors[training_slice:training_slice+test_slice]
l2_test_data = l2_vectors[training_slice:training_slice+test_slice]

l1_seed_data = l1_vectors[training_slice+test_slice:training_slice +
                          test_slice+seed_slice]
l2_seed_data = l2_vectors[training_slice+test_slice:training_slice +
                          test_slice+seed_slice]

# Instantiates one VectorSpace object for each of the languages, to train on
l1_vs_train = VectorSpace(l1_training_data)
l2_vs_train = VectorSpace(l2_training_data)

# Instantiates one VectorSpace object for each of the languages, to test on
l1_vs_test = VectorSpace(l1_test_data)
l2_vs_test = VectorSpace(l2_test_data)

# Instantiates one VectorSpace object for each of the languages, to use as seed
# dictionary
l1_seed = VectorSpace(l1_seed_data)
l2_seed = VectorSpace(l2_seed_data)

# Instantiates a Translator object with the seed data
translator = Translator([l1_seed, l2_seed])
# Trains the Translator object with the training data
translator.train(l1_vs_train, l2_vs_train)
# Evaluates the trained Translator object using the test data
translator.evaluate(l1_vs_test, l2_vs_test)
