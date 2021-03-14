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


class TextGCNGraphBuilder(implements(GraphBuilderInterface)):

    def build_graph(
        self,
        documents: List[Document]
    ) -> List[GraphMatrix]:
        raise NotImplementedError

    def _build_adjacency_matrix(self) -> np.array:
        raise NotImplementedError

    def _build_feature_matrix(self) -> np.array:
        raise NotImplementedError
