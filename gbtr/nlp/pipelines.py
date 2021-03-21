from interface import Interface, implements
from typing import List
from .processors import ProcessorInterface


class ProcessingPipelineInterface(Interface):
    """Class for sequential text processing with given NLP steps"""

    def __init__(
        self,
        steps: List[ProcessorInterface]
    ) -> None:
        pass

    def process(
        self,
        text: str
    ) -> str:
        pass


class ProcessingPipeline(implements(ProcessingPipelineInterface)):

    def __init__(
        self,
        steps: List[ProcessorInterface]
    ) -> None:

        self.steps = steps

    def process(
        self,
        text: str
    ) -> str:

        for step in self.steps:
            text = step.process(text)

        return text
