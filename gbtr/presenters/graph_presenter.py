import networkx as nx

from ..model.graph_matrix import GraphMatrix


class GraphPresenter:
    """Class with methods to transform graph matrix to selected form."""

    def to_nx(
        self,
        graph_matrix: GraphMatrix
    ) -> nx.Graph:
        """Transform graph matrix to networkx graph object."""
        raise NotImplementedError
