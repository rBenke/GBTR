from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class GraphMatrix:
    """Class for storing graph information in matrix form.

    Numpy array is used as standard matrix interface.

    Parameters
    ----------
    adjacency_matrix : np.array
    nodes_features_matrix : np.array, optional
    edges_features_matrix : np.array, optional
    label: str, optional
        Label field using if graph represents single document.
    labels: dict, optional
        Dictionary {document_id: str : label: str}
        Labels dict using if graph represents many documents.
        Document id is equal node id in matrix.
    """

    adjacency_matrix: np.array
    nodes_features_matrix: np.array = None
    edges_features_matrix: np.array = None
    label: str = None
    labels: Dict[int: str]
