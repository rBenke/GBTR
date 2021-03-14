from dataclasses import dataclass


@dataclass
class Document:
    """Class for storing data about single text document.

    Parameters
    ----------
    text : str
        Text data.
    label : str, optional
        Document label.
    name : str, optional
        Document name.
    """

    text: str
    label: str = None
    name: str = None
