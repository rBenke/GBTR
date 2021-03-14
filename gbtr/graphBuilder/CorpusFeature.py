import abc
class CorpusFeatureInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'get_feature') and
                callable(subclass.get_feature) or
                NotImplemented)

    @abc.abstractmethod
    def get_feature(self, documents: list):
        """Prepare feature vector for every document in the corpus"""
        raise NotImplementedError


