from typing import List, Dict

from interface import Interface, implements

from ..model.document import Document


class DictReaderInterface(Interface):
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
        pass


class DictReader(implements(DictReaderInterface)):

    def read_data(
        self,
        data: List[Dict[str, str]]
    ) -> List[Document]:

        raise NotImplementedError
