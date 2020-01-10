import sys
from collections import defaultdict
import pickle
import time

word_lexicon = {}
l2_word_lexicon = {}

l1_vectors = {}
l2_vectors = {}

l1_vec = []
l2_vec = []


def make_word_lexicon():
    """Populates l1_word_lexicon and l2_word_lexicon with their words and
    respective translations."""
    t0 = time.time()
    with open(sys.argv[1]) as f1:
        for line in f1:
            line = line.split()
            word_lexicon.setdefault(line[0], line[1])
            l2_word_lexicon.setdefault(line[1], line[0])
    t1 = time.time()
    print('make_word_lexicon', t1-t0)
                
def l1_vec_lexicon(): 
    """If we have a translation for the l1-vector, adds the vector to
    l1_vectors."""
    t0 = time.time()
    with open(sys.argv[2]) as f2:
        next(f2)
        for line in f2:
            line = line.split()
            if line[0] in word_lexicon:
                l1_vectors[line[0]] = line[1:]
            else:
                pass
    t1 = time.time()
    print('l1_vec_lexicon', t1-t0)

def l2_vec_lexicon(): # takes a VERY long time
    """If we have a translation for the l2-vector, adds the vector to
    l2_vectors."""
    t0 = time.time()
    with open(sys.argv[3]) as f3:
        next(f3)
        for line in f3:
            line = line.split()
            if line[0] in l2_word_lexicon:
                l2_vectors[line[0]] = line[1:]
            else:
                pass
    t1 = time.time()

    print('l2_vec_lexicon', t1-t0)

def match_vectors():
    """If we have both of the vectors for an l1-l2 pair, adds them in
    parallell to each of l1_vec and l2_vec."""
    
    t0 = time.time()

    for k,v in word_lexicon.items():
        if k in l1_vectors and v in l2_vectors:
            l1_vec.append([k, l1_vectors[k]]) 
            l2_vec.append([v, l2_vectors[v]])
    t1 = time.time()

    print('match_vectors', t1-t0)


make_word_lexicon()
l1_vec_lexicon()
l2_vec_lexicon()
match_vectors()

# saves each of the lists as a pickle dump

with open('l1_vec.pkl', 'wb') as f:
    pickle.dump(l1_vec, f)

with open('l2_vec.pkl', 'wb') as f:
    pickle.dump(l2_vec, f)
print(len(l1_vec))
print(len(l2_vec))
print('done')
