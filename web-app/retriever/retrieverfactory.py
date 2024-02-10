from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore

from haystack_integrations.components.retrievers.opensearch import OpenSearchBM25Retriever, OpenSearchEmbeddingRetriever


class BaseRetrieverFactory:
    def create_retriever(self, document_store: OpenSearchDocumentStore) :
        raise NotImplementedError("Subclasses must implement create_retriever method")


class BM25RetrieverFactory(BaseRetrieverFactory):
    def create_retriever(self, document_store: OpenSearchDocumentStore) -> OpenSearchBM25Retriever:
        return OpenSearchBM25Retriever(document_store=document_store)


class EmbeddingRetrieverFactory(BaseRetrieverFactory):
    def create_retriever(self, document_store: OpenSearchDocumentStore) -> OpenSearchEmbeddingRetriever:
        return OpenSearchEmbeddingRetriever(document_store=document_store)
