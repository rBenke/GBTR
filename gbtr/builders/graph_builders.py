from typing import List

from interface import Interface, implements
import numpy as np

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

        words_order = np.unique(self.documents)

        word_word_adj = self._word_word_matrix(words_order)
        doc_word_adj = self._doc_word_matrix(words_order)
        word_doc_adj = doc_word_adj.T
        doc_doc_adj = self._doc_doc_matrix(words_order)

        #              word_word_adj | word_doc_adj
        #  Adj_mat =   -----------------------------
        #              doc_word_adj  | doc_doc_adj

        col1 = np.row_stack((word_word_adj, doc_word_adj))
        col2 = np.row_stack((word_doc_adj, doc_doc_adj))
        graph_adj = np.column_stack((col1, col2))

        return graph_adj

    def _build_feature_matrix(self) -> np.array:
        raise NotImplementedError

    def _doc_doc_matrix(self,words_order):
        raise NotImplementedError

    def _doc_word_matrix(self,words_order):
        from sklearn.feature_extraction.text import TfidfVectorizer

        tfidf = TfidfVectorizer(vocabulary=words_order)
        filtered_text = [" ".join(text) for text in tokenized_texts]
        doc_word_adj = tfidf.fit_transform(filtered_text)
        return doc_word_adj

    def _word_word_matrix(self, words_order):
        from nltk.util import ngrams
        import pandas as pd

        unigram_prob = np.unique(self.documents, return_counts=True)


        unigram_prob_matrix = np.matmul(np.expand_dims(unigram_prob, 1), np.expand_dims(unigram_prob, 1).T)

        bigrams = [list(ngrams(text, 2)) for text in self.documents]
        bigrams = list(itertools.chain.from_iterable(bigrams))

        bigram_freq = pd.value_counts(bigrams)

        bigram_freq_set = {}
        for a, b in zip(bigram_freq.index, bigram_freq):
            bigram_freq_set[a] = int(b)

        bigram_matrix = np.zeros((len(words_order), len(words_order)))

        for i in range(len(words_order)):
            word1 = words_order[i]
            for j in range(i + 1, len(words_order)):
                word2 = words_order[j]
                try:
                    bi_coo1 = bigram_freq_set[(word1, word2)]
                except:
                    bi_coo1 = 0
                try:
                    bi_coo2 = bigram_freq_set[(word2, word1)] + bi_coo1
                except:
                    bi_coo2 = 0 + bi_coo1

                bigram_matrix[i, j] = bi_coo2
                bigram_matrix[j, i] = bi_coo2

        bigram__prob_matrix = bigram_matrix / float(sum(bigram_freq))

        adj_matrix = np.log(bigram__prob_matrix / unigram_prob_matrix)
        adj_matrix[adj_matrix < 0] = 0
        return (adj_matrix)


