from abc import ABC, abstractmethod


class Retriever(ABC):
    @abstractmethod
    def retrieve_docs(self, query, k=5):
        pass


