from .baseretriever import Retriever
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_community.embeddings import OllamaEmbeddings


class OpenSearchRetriever(Retriever):
    def __init__(self):
        embeddings = OllamaEmbeddings(model="llama2:7b")
        self.docsearch = OpenSearchVectorSearch(
            opensearch_url="http://localhost:9200",
            index_name="wiki",
            embeddings=embeddings,
        )

    def retrieve_docs(self, query, k=5):
        docs = self.docsearch.similarity_search(
            query,
            search_type="script_scoring",
            space_type="cosinesimil",
            vector_field="message_embedding",
            text_field="message",
            metadata_field="message_metadata",
        )
        return docs
