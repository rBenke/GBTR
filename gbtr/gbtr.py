from typing import Any, List

import networkx as nx
from interface import Interface, implements

from .builders.graph_builders import GraphBuilderInterface, TextGCNGraphBuilder
from .model.document import Document
from .model.graph_matrix import GraphMatrix
from .nlp.pipelines import ProcessingPipeline, ProcessingPipelineInterface
from .nlp.processors import Lemmatizer
from .presenters.graph_presenter import GraphPresenter
from .readers.reading_controller import ReadingController


class GBTRInterface(Interface):
    """Main module."""

    def get_graph(
        self,
        source: Any
    ) -> List[GraphMatrix]:
        """Transform given documents corpus to graph representation.

        Parameters
        ----------
        source: any
            Data source in one of supported types.
            Currently supported types:
            - list of dictionaries {"text" : str, "label" : str}.

        Returns
        -------
        List[GraphMatrix]
            List of prepared graphs.
            If method implements whole corpus representation as one graph
            then one element list is returned.
        """


class GBTR(implements(GBTRInterface)):

    def __init__(
        self,
        reading_controller: ReadingController,
        nlp_pipeline: ProcessingPipelineInterface,
        graph_builder: GraphBuilderInterface
    ):

        self._data: List[Document] = None
        self._reading_controller = reading_controller
        self._graph_builder = graph_builder

    def get_graph(
        self,
        source: Any
    ) -> List[GraphMatrix]:

        self._data = self._reading_controller.read_data(source)

        # TODO
        # consider parallel processing
        for document in self._data:
            document.text = self.nlp_pipeline.process(document.text)

        return self._graph_builder.get_graph(self._data)


class TextGCN:
    """Implementation of graph representation for TextGCN."""

    def __call__(
        self,
        source: Any
    ) -> nx.Graph:
        """Returns TextGCN based grapg representation for given corpus.

        Parameters
        ----------
        source: any
            Data source in one of supported types.
            Currently supported types:
            - list of dictionaries {"text" : str, "label" : str}.

        Returns
        -------
        nx.Graph
            Graph representation as Networkx Graph object.
        """

        gbtr = GBTR(
            reading_controller=ReadingController(),
            nlp_pipeline=ProcessingPipeline([
                # TODO
                Lemmatizer()
            ]),
            graph_builder=TextGCNGraphBuilder()
        )

        graph_matrix = gbtr.get_graph(source)[0]

        graph_presenter = GraphPresenter()
        return graph_presenter.to_nx(graph_matrix)
