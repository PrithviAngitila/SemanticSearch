from .baseretriever import Retriever
from haystack_integrations.components.retrievers.opensearch import OpenSearchBM25Retriever
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore


class BM25Retriever(Retriever):
    def __init__(self):
        self.document_store = OpenSearchDocumentStore(
            hosts="http://localhost:9200", use_ssl=False, verify_certs=False, index='wiki'
        )
        self.retriever = OpenSearchBM25Retriever(document_store=self.document_store)

    def retrieve_docs(self, query, k=5, index_name="wiki"):
        return self.retriever.retrieve(query=query, top_k=k, index=index_name)
