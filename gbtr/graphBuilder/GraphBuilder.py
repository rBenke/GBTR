import abc
class GraphBuilderInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'build_graph') and
                callable(subclass.build_graph) and
                hasattr(subclass, 'build_adjacency_matrix') and
                callable(subclass.build_adjacency_matrix) and
                hasattr(subclass, 'build_feature_matrix') and
               callable(subclass.build_feature_matrix) or
                NotImplemented)

    @abc.abstractmethod
    def build_graph(self, documents: list):
        """Build a graph from list of documents"""
        raise NotImplementedError

    @abc.abstractmethod
    def build_adjacency_matrix(self):
        """Private method - prepare adjacency matrix"""
        raise NotImplementedError

    @abc.abstractmethod
    def build_feature_matrix(self):
        """Private method - prepare feature matrix"""
        raise NotImplementedError

