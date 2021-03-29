from typing import List

from interface import Interface, implements
import numpy as np
from nltk.util import ngrams
from corpus_features import TF_IDF
from ..model.document import Document
from ..model.graph_matrix import GraphMatrix


class GraphBuilderInterface(Interface):

    def build_graph(
            self,
            documents: List[Document]
    ) -> List[GraphMatrix]:
        """Build a graph from list of documents"""
        pass

    def _build_adjacency_matrix(
            self
    ) -> np.array:
        """Build adjacency matrix"""
        pass

    def _build_feature_matrix(
            self
    ) -> np.array:
        """Build feature matrix"""
        pass


class TextGCNGraphBuilder(implements(GraphBuilderInterface)):
    """
    TextGCN is a method introduced by Liang Yao, Chengsheng Mao, Yuan Luo in
    "Graph Convolutional Networks for Text Classification". The whole corpuse is mapped
    to a single heterogenoues graph with two types of nodes: document nodes and word nodes.

    How does the adjacency matrix looks like?
    Two words, let's say word1 and word2, are connected with each with the weight max( PMI(word1,word2), 0 ).
    Documents with words are connected based on the TF-IDF.
    Document is never connected with another document directly.

    How does the feature matrix looks like?
    Feature matrix is an identity matrix with the size of all nodes.
    """

    def build_graph(
            self,
            documents: List[Document]
    ) -> List[GraphMatrix]:

        adjacency_matrix = self._build_adjacency_matrix()
        feature_matrix = self._build_feature_matrix()

        labels_list = [(idx, document.label) for idx, document in enumerate(self.documents)]
        labels_dict = dict(labels_list)

        TextGCN_gm = GraphMatrix(
            adjacency_matrix=adjacency_matrix,
            nodes_features_matrix=feature_matrix,
            labels=labels_dict)

        return [TextGCN_gm]

    def _build_adjacency_matrix(self) -> np.array:

        words_order = np.unique(self.documents, return_counts=True)

        word_word_adj = self._word_word_matrix(words_order[0])
        doc_word_adj = self._doc_word_matrix(words_order)
        word_doc_adj = doc_word_adj.T
        doc_doc_adj = self._doc_doc_matrix(words_order[0])

        #              word_word_adj | word_doc_adj
        #  Adj_mat =   -----------------------------
        #              doc_word_adj  | doc_doc_adj

        col1 = np.row_stack((word_word_adj, doc_word_adj))
        col2 = np.row_stack((word_doc_adj, doc_doc_adj))
        graph_adj = np.column_stack((col1, col2))

        return graph_adj


    def _doc_doc_matrix(self, words_order):
        matrix_size = (len(self.documents), len(self.documents))
        return np.zeros((matrix_size))

    def _doc_word_matrix(self, words_order):
        doc_word_adj = TF_IDF(self.documents, words_order)
        return doc_word_adj

    def _word_word_matrix(self, words_order):

        unigrams_freq = words_order[1]/np.sum(words_order[1])

        unigram_prob_matrix = np.matmul(np.expand_dims(unigrams_freq, 1), np.expand_dims(unigrams_freq, 1).T)

        bigrams = [ngrams(text, 2) for text in self.documents]
        bigram_uniq = np.unique(bigrams, return_counts=True)

        bigram_matrix = np.zeros((len(words_order), len(words_order)))
        unigrams_position_lookup = dict(zip(words_order[0], len(words_order[0])))

        for i in range(len(bigram_uniq[0])):
            word1 = bigram_uniq[0][i][0]
            word1pos = unigrams_position_lookup[word1]
            word2 = bigram_uniq[0][i][1]
            word2pos = unigrams_position_lookup[word2]
            count = bigram_uniq[1][i]
            bigram_matrix[word1pos, word2pos] = bigram_matrix[word1pos, word2pos] + count
            bigram_matrix[word2pos, word1pos] = bigram_matrix[word2pos, word1pos] + count

        bigram_prob_matrix = bigram_matrix / float(sum(bigram_uniq[0]))

        # PMI
        adj_matrix = np.log(bigram_prob_matrix / unigram_prob_matrix)
        adj_matrix[adj_matrix < 0] = 0

        np.fill_diagonal(adj_matrix,0)

        return adj_matrix

    def _build_feature_matrix(self, words_order) -> np.array:
        matrix_shape = (len(words_order) + len(self.documents), len(words_order) + len(self.documents))
        return np.ones(matrix_shape)
