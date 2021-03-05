from abc import ABC, abstractmethod
from typing import Dict, List

from ..model.document import Document


class DictReaderInterface(ABC):

    @abstractmethod
    def read_data(
        self,
        data: List[Dict[str, str]]
    ) -> List[Document]:
        pass


class DictReader(DictReaderInterface):
    """Reader for dictionary like data source."""

    def read_data(
        self,
        data: List[Dict[str, str]]
    ) -> List[Document]:
        """Reads data from given dictionary

        Parameters
        ----------
        source: data
            List of documents as dictiionaries {"text" : str, "label" : str}.

        Returns
        -------
        List[Document]
            List collection of Document instances.
        """

    raise NotImplementedError
