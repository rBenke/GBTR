from interface import Interface, implements


class ProcessorInterface(Interface):
    """Class for NLP methods that process string to string."""

    def process(
        self,
        text: str
    ) -> str:
        """Process and return given string."""
        pass


class Lemmatizer(implements(ProcessorInterface)):

    def process(
        self,
        text: str
    ) -> str:
        raise NotImplementedError
