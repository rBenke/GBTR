from typing import List

from interface import Interface, implements
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..model.document import Document


class CorpusFeatureInterface(Interface):

    def get_feature(
        self,
        documents: List[Document]
    ) -> np.array:
        """Prepare feature vector for every document in the corpus"""
        pass


class TF_IDF(implements(CorpusFeatureInterface)):

    def get_feature(
        self,
        documents: List[Document], words_order
    ) -> np.array:

        tfidf = TfidfVectorizer(vocabulary=words_order)
        filtered_text = [" ".join(text) for text in documents]
        tfidf.fit_transform(filtered_text)
