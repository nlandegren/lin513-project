from preprocessor import Preprocessor
from classifier import Classifier
import sys
import time
import bz2
import os


def main():

    path = sys.argv[1]
    file_names = []
    for root, dir_names, f_names in os.walk(path):
        for f in f_names:
            file_names.append(os.path.join(root, f))

    pre = Preprocessor(file_names, window_size = 10)
    classif = Classifier(len(pre.word_index))

    make_embeddings(file_names, pre, classif)


def make_embeddings(file_names, pre, classifier):

    start = time.time()
    for f in file_names:
        with bz2.open(f, 'rb') as fin:
            for line in fin:
                line = line.decode()
                if not line.startswith('<') and len(line) > 1:
                    line = line.split()
                    line = [pre[word] for word in line]
                    print(line)
                    skipgrams = pre.make_skipgrams(line)

                    for i, word in enumerate(line):
                        real_context = pre.positive_context(i, skipgrams)
                        if (word != None and pre.subsample(word) and None not
                        in real_context):
                            fake_context = pre.negative_context(word)
                            classifier.train(word, real_context, fake_context)
    
    end = time.time()
    print('training time: ', end - start)
    with open(sys.argv[2]+'.vec', 'w') as fout:
        for i, vec in enumerate(classifier.target_matrix):
            vec = str(list(vec))
            fout.write(intex_to_word[i]+' '+vec+'\n')


if __name__=='__main__':
    main()
