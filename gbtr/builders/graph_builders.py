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

    def build_graph(
        self,
        documents: List[Document]
    ) -> List[GraphMatrix]:

        adjacency_matrix = self._build_adjacency_matrix()
        feature_matrix = self._build_feature_matrix()

        labels_list = [(idx, document.label) for idx, document in enumerate(self.documents)]
        labels_dict = dict(labels_list)

        TextGCNGraph = GraphMatrix(
            adjacency_matrix=adjacency_matrix,
            nodes_features_matrix=feature_matrix,
            labels=labels_dict)

        return list(TextGCNGraph)


    def _build_adjacency_matrix(self) -> np.array:
        import nltk
        from nltk import word_tokenize
        from nltk.util import ngrams
        X_tokenized = [word_tokenize(text.lower()) for text in X]

        word_word_adj = self._word_word_matrix(tokenized_corpus, words_order)
        doc_word_adj = self._doc_word_matrix(tokenized_corpus, words_order)
        word_doc_adj = doc_word_adj.T
        doc_doc_adj = self._doc_doc_matrix(tokenized_corpus, words_order)

        #              word_word_adj | word_doc_adj
        #  Adj_mat =   -----------------------------
        #              doc_word_adj  | doc_doc_adj

        col1 = np.row_stack((word_word_adj, doc_word_adj))
        col2 = np.row_stack((word_doc_adj, doc_doc_adj))
        graph_adj = np.column_stack((col1, col2))
        graph_adj_df = pd.DataFrame(graph_adj, columns=useful_tokens + list(range(Y.shape[0])),
                                    index=useful_tokens + list(range(Y.shape[0])))
        return graph_adj_df.to_numpy()

    def _build_feature_matrix(self) -> np.array:
        raise NotImplementedError

    def _doc_doc_matrix(tokenized_texts, words_order):
        raise NotImplementedError
    def _doc_word_matrix(tokenized_texts, words_order):
        from sklearn.feature_extraction.text import TfidfVectorizer

        tfidf = TfidfVectorizer(vocabulary=words_order)
        filtered_text = [" ".join(text) for text in tokenized_texts]
        doc_word_adj = tfidf.fit_transform(filtered_text)
        return doc_word_adj

    def _word_word_matrix(tokenized_texts, words_order):
        unigram_freq = pd.value_counts(list(itertools.chain.from_iterable(tokenized_texts)))

        unigram_prob = unigram_freq / float(sum(unigram_freq))
        unigram_prob_matrix = np.matmul(np.expand_dims(unigram_prob, 1), np.expand_dims(unigram_prob, 1).T)

        bigrams = [list(ngrams(text, 2)) for text in tokenized_texts]
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


