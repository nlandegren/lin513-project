from preprocessor import Preprocessor
from classifier import Classifier
import sys
import bz2
import os


def main():
    file_names = get_files()
    pre = Preprocessor(file_names, window_size=10)
    classif = Classifier(len(pre.word_index))
    embeddings = make_embeddings(file_names, pre, classif)
    write_to_file(embeddings)


def get_files():
    path = sys.argv[1]
    file_names = []
    for root, dir_names, f_names in os.walk(path):
        for f in f_names:
            file_names.append(os.path.join(root, f))
    return file_names


def make_embeddings(file_names, pre, classifier):
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
                    # Only train if word is in top 200k most common words, it's skipgram only contains 200k most common words and it's not subsampled
                    if (word is not None and pre.subsample(word) and None
                       not in real_context):
                        fake_context = pre.negative_context(word)
                        classifier.train(word, real_context, fake_context)
    return classifier.context_matrix, pre.index_word


def write_to_file(out_data):
    with open(sys.argv[2]+'.vec', 'w') as fout:
        for i, vec in enumerate(out_data[0]):
            vec = ''.join(list(vec))
            fout.write(out_data[1][i]+' '+vec+'\n')


if __name__ == '__main__':
    main()
