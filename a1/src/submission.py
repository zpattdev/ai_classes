#!/usr/bin/env python

import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils import *
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

sys.path.append(os.path.abspath(os.path.join('..')))

def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): list of distinct words across the corpus, sorted (using python 'sorted' function)
            num_corpus_words (integer): number of distinct words across the corpus
    """
    corpus_words = []
    num_corpus_words = 0

    # ### START CODE HERE ###
    corpus_words = sorted(list(set([word for col in corpus for word in col])))
    num_corpus_words = len(corpus_words)
    # ### END CODE HERE ###

    return corpus_words, num_corpus_words

def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).

        Note: Each word in a document should be at the center of a window. Words near edges will have a smaller
              number of co-occurring words.

              For example, if we take the document "START All that glitters is not gold END" with window size of 4,
              "All" will co-occur with "START", "that", "glitters", "is", and "not".

        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (numpy matrix of shape (number of unique words in the corpus , number of unique words in the corpus)):
                Co-occurrence matrix of word counts.
                The ordering of the words in the rows/columns should be the same as the ordering of the words given by the distinct_words function.
            word2Ind (dict): dictionary that maps word to index (i.e. row/column number) for matrix M.
    """
    def _get_window(center_idx, line):
        start_idx = max(0, center_idx - window_size)
        end_idx = min(len(line), center_idx + window_size + 1)
        return line[start_idx:end_idx]

    words, num_words = distinct_words(corpus)
    M = np.zeros([num_words, num_words], dtype='float')
    word2Ind =  {word: idx for idx, word in enumerate(words)}

    # ### START CODE HERE ###
    for line in corpus:
        for idx, center_word in enumerate(line):
            for context_word in _get_window(idx, line):
                if context_word == center_word:
                    continue
                M[word2Ind[context_word], word2Ind[center_word]] += 1.0
    # ### END CODE HERE ###

    return M, word2Ind

def compute_co_occurrence_matrix_dense(corpus, window_size=4):
    """
    Dense, efficient co-occurrence matrix.
    Returns: M (numpy array float shape (V,V)), word2Ind (dict)
    """
    words, _ = distinct_words(corpus)
    word2Ind = {w:i for i,w in enumerate(words)}
    V = len(words)
    M = np.zeros((V, V), dtype=float)

    # local var binding for speed
    wsize = int(window_size)
    idx_map = word2Ind
    mat = M

    for doc in corpus:
        L = len(doc)
        # iterate token by token
        for i, center in enumerate(doc):
            ci = idx_map.get(center)
            if ci is None:
                continue
            # window bounds (exclude center)
            left = max(0, i - wsize)
            right = min(L, i + wsize + 1)  # python slice end-exclusive
            # iterate neighbors
            # note: skip center at position i
            for j in range(left, right):
                if j == i:
                    continue
                neighbor = doc[j]
                nj = idx_map.get(neighbor)
                if nj is None:
                    continue
                mat[ci, nj] += 1.0

    return M, word2Ind

def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurrence count matrix of dimensionality (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using the following SVD function from Scikit-Learn:
            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html

        Params:
            M (numpy matrix of shape (number of unique words in the corpus , number of unique words in the corpus)): co-occurrence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (number of unique words in the corpus, k)): matrix of k-dimensioal word embeddings.
                    In terms of the SVD from math class, this actually returns U * S
    """
    np.random.seed(4355)
    n_iter = 10     # Use this parameter in your call to `TruncatedSVD`
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))

    # ### START CODE HERE ###
    svd = TruncatedSVD(n_components=k, n_iter=n_iter)
    M_reduced = svd.fit_transform(M)
    
    # ### END CODE HERE ###

    print("Done.")
    return M_reduced

def main():
    matplotlib.use('agg')
    plt.rcParams['figure.figsize'] = [10, 5]

    assert sys.version_info[0] == 3
    assert sys.version_info[1] >= 5

    def plot_embeddings(M_reduced, word2Ind, words, title):

        for word in words:
            idx = word2Ind[word]
            x = M_reduced[idx, 0]
            y = M_reduced[idx, 1]
            plt.scatter(x, y, marker='x', color='red')
            plt.text(x, y, word, fontsize=9)
        plt.savefig(title)

    #Read in the corpus
    reuters_corpus = read_corpus()

    M_co_occurrence, word2Ind_co_occurrence = compute_co_occurrence_matrix(reuters_corpus)
    M_reduced_co_occurrence = reduce_to_k_dim(M_co_occurrence, k=2)
    # Rescale (normalize) the rows to make them each of unit-length
    M_lengths = np.linalg.norm(M_reduced_co_occurrence, axis=1)
    M_normalized = M_reduced_co_occurrence / M_lengths[:, np.newaxis] # broadcasting

    words = ['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela']
    plot_embeddings(M_normalized, word2Ind_co_occurrence, words, 'co_occurrence_embeddings_(soln).png')

if __name__ == "__main__":
    main()
