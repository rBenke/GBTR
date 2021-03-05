from typing import Any, List

from ..model.document import Document
from .dict_readers import DictReaderInterface, DictReader


class ReadingController:
    """Class contains and selects proper data reader."""

    @property
    def dict_reader() -> DictReaderInterface:
        """DictReaderInterface: instanfe of dictionary reader"""
        return DictReader()

    def read_data(
        self,
        source: Any
    ) -> List[Document]:
        """Selects proper reader and reads data.

        Parameters
        ----------
        source: any
            Data source in one of supported types.
            Currently supported types:
            - list of dictionaries {"text" : str, "label" : str}.

        Returns
        -------
        List[Document]
            List collection of Document instances.
        """

        raise NotImplementedError
