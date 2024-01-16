from .baseretriever import Retriever
from opensearch_haystack import OpenSearchBM25Retriever
from opensearch_haystack import OpenSearchDocumentStore


class BM25Retriever(Retriever):
    def __init__(self):
        self.document_store = OpenSearchDocumentStore(
            hosts="http://localhost:9200", use_ssl=False, verify_certs=False, index='haystack'
        )
        self.retriever = OpenSearchBM25Retriever(document_store=self.document_store)

    def retrieve_docs(self, query, k=5, index_name="haystack"):
        return self.retriever.retrieve(query=query, top_k=k, index=index_name)
