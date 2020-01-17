"""A main module acting as a user interface for the embedding program.

The script uses a Preprocessor object to convert text into training data with
which to train a Classifier object. The input to the script is expected to be
the pathway to a directory containing the text files with the training data.

    Usage:
    python3 main.py path_to_dir
"""

from preprocessor import Preprocessor
from classifier import Classifier
import sys
import os


def main():
    """The main function of the module.

    Gets training data from provided files and trains a Classifier object with
    the help of a Preprocessor object, then writes the resulting embeddings to
    file.
    """
    file_names = get_files(sys.argv[1])
    pre = Preprocessor(file_names, window_size=10)
    classif = Classifier(len(pre.word_index))
    out_data = make_embeddings(file_names, pre, classif)
    write_to_file(sys.argv[2], out_data)


def get_files(dir_path):
    """Makes a list of all the files to read through.
    Args:
        dir_path: The path to a directory containing the files to read.
    Returns:
        A list of the pathways leading to each file in dir_path, each as str
        objects.
    """
    file_names = []
    # Populates the list file_names with the path to each text file in the dir
    for root, dir_names, f_names in os.walk(dir_path):
        for f in f_names:
            file_names.append(os.dir_path.join(root, f))
    return file_names


def make_embeddings(file_names, pre, classifier):
    """Walks through data and trains a Classifier object.
    Args:
        file_names: A list of pathways to text files.
        pre: A Preprocessor object.
        classifier: A Classifier object.
    Returns:
        A tuple of the target matrix of the trained Classifier object, and an
        index to word dict object.
    """
    for i, f in enumerate(file_names):
        print('{} out of {} files trained on'.format(i+1, len(file_names)))
        with open(f, 'r') as fin:
            for line in fin:
                # Converts each word to it's unique index
                line = [pre[word] for word in line.split()]
                if len(line) < 5 or None in line:
                    continue
                # Gets the skipgram for each word in the line
                skipgrams = pre.make_skipgrams(line)
                for i, word in enumerate(line):
                    # Picks out a skipgram for the given word to act as it's
                    # positive training example
                    real_context = pre.positive_context(i, skipgrams)
                    # Only train on 200k most common words and according to
                    # subsampling
                    if pre.subsample(word):
                        fake_context = pre.negative_context(word)
                        classifier.train(word, real_context, fake_context)
    return classifier.target_matrix, pre.index_word


def write_to_file(file_name, out_data):
    """Writes results to a text file.
    Args:
        out_file_name: The name of the file to be written to.
        out_data: The data returned from make_embeddings function, a 2-tuple
        containing a numpy ndarray containing trained word embeddings and a
        dict object associating each word embedding index as an int with their
        actual words as str objects.
    """
    with open(file_name+'.vec', 'w') as fout:
        # Writes meta info to first row, number of embeddings and dimensions
        fout.write(str(len(out_data[0]))+' '+str(len(out_data[0][0]))+'\n')
        # Writes each word together with it's vector to text file
        for i, vec in enumerate(out_data[0]):
            vec = ' '.join(list(str(f) for f in vec))
            fout.write(out_data[1][i]+' '+vec+'\n')


if __name__ == '__main__':
    main()
