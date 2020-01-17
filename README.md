# lin513-project

### Description
This project was apart of a course in applied programming for linguists at Stockholm University. The goal of the project was to implement a somewhat more complex program in python to solve a proplem in the area of computational linguistics. The purpose was to develop skills in structuring and documenting code in a readable and efficient way. I built two programs, one for implementing a machine translation algorithm as described by Artetxe et.  al.  2017, and a second program implementing the skipgram word2vec algorithm for word embedding [Mikolov et.  al.  2013].

The two programs consists of a directory each: 'embedding' and 'translation'. The directory 'embedding' contains several python scripts used to create and test word embeddings. The directory 'translation' contains python scripts used to train and evaluate the translation model. 

---

### Clone
Clone this repo to your local machine using:
```
$ git clone https://github.com/nlandegren/lin513-project
```
---

### Usage
Run the code from your terminal as follows.

To train word embeddings:
```
$ python3 embedding/main.py dir_containing_text_files outfile_name
```
To do a simple similarity test on your word embeddings:
```
$ python3 embedding/similarity_test.py vector_file.vec
```
To parallel-sort two vector files against a lexicon:
```
$ python3 embedding/make_parallel.py l1-l2-lexicon.txt l1_vector_file.vec l2_vector_file.vec
```
To test the translation model:
```
$ python3 translation/main.py l1_vector_file l2_vector_file
```
---
### Documentation

embedding/
> embedding/preprocessor.py\
This module contains the Preprocessor class. It prepares the data needed to train a classifier with the word2vec skipgram algorithm.

> embedding/classifier.py\
This module contains the Classifier class which performs the fake tast of classifying sets of words as real or fake contexts of a give target word. The parameters of the trained Classifier object is then used as word embeddings.

> embedding/main.py\
The main module of the 'embedding' directory. Uses instances of the Preprocessor class and Classifier class to make word embeddings.

> embedding/similarity_test.py\
A short script that prints out the five most similar word embeddings for a few random embeddings in a vector file.

> embedding/make_parallel.py\
A script that will take two files of word embeddings and output the embeddings sorted in parallel into new files.

translation/

> translation/vectorspace.py\
This module contains the VectorSpace class, which acts as a wrapper for a set of word embeddings. Instances of the class are used in training and testing of the translation model.

> translation/translator.py\
This module contains the Translator class. The class represents a machine translation model based on singular value decomposition.

> translation/main.py\
The main module of the translation program. Uses instances of the VectorSpace class and Translator class to train and test the correctness of the translation algorithm.
---

### References
- Mikolov, T.; Chen, K.; Corrado, G.; Dean, J. 2013. Efficient estimation of word representations in vector space.
arXiv preprint arXiv:1301.3781.
- Artetxe, M.; Labaka, G.; Agirre, E. 2017. Learning bilingual word embeddings with (almost) no bilingual data. 
Proceedings of the 55th annual meeting of the association for computational linguistics (volume 1:  Long papers) (451–462)
---
### License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

**[MIT license](http://opensource.org/licenses/mit-license.php)**
