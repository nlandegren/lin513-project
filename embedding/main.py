from preprocessor import Preprocessor
from classifier import Classifier
import sys
import os


def main():
    """The main function of the module.

    Gets training data from provided files and trains a classifier object with
    the help of a Preprocessor object.
    """
    file_names = get_files()
    pre = Preprocessor(file_names, window_size=10)
    classif = Classifier(len(pre.word_index))
    embeddings = make_embeddings(file_names, pre, classif)
    write_to_file(embeddings)


def get_files():
    # The path to a directory
    path = sys.argv[1]
    file_names = []
    # Populates file_names with the path to each text file in the dir
    for root, dir_names, f_names in os.walk(path):
        for f in f_names:
            file_names.append(os.path.join(root, f))
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
    for f in file_names:
        with open(f, 'r') as fin:
            for line in fin:
                # Converts each word to it's unique index
                line = [pre[word] for word in line.split()]
                # Gets the skipgram for each word in the line
                skipgrams = pre.make_skipgrams(line)

                for i, word in enumerate(line):
                    # Picks out a skipgram for the given word to act as it's
                    # positive training example
                    real_context = pre.positive_context(i, skipgrams)
                    # Only train on 200k most common words and according to
                    # subsampling
                    if (word is not None and None not in real_context
                            and pre.subsample(word)):
                        fake_context = pre.negative_context(word)
                        classifier.train(word, real_context, fake_context)
    return classifier.target_matrix, pre.index_word


def write_to_file(out_data):
    """Writes results to a text file."""
    with open(sys.argv[2]+'.vec', 'w') as fout:
        # Writes each word together with it's vector to text file
        for i, vec in enumerate(out_data[0]):
            vec = ''.join(list(vec))
            fout.write(out_data[1][i]+' '+vec+'\n')


if __name__ == '__main__':
    main()
