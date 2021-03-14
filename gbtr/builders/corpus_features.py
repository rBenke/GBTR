from typing import List

from interface import Interface, implements
import numpy as np

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
        documents: List[Document]
    ) -> np.array:
        raise NotImplementedError
