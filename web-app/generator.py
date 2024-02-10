from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
from retriever import BM25RetrieverFactory, EmbeddingRetrieverFactory
from retriever import BM25RetrieverWrapper, EmbeddingRetrieverWrapper
from haystack_integrations.components.generators.ollama import OllamaGenerator


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    

class Generator(metaclass=Singleton):
    def __init__(self, retriever_type="bm25"):
        self.document_store = OpenSearchDocumentStore(
            hosts="http://localhost:9200",
            use_ssl=False,
            verify_certs=False,
            index="wiki",
        )
        self.llm = OllamaGenerator(
            model="llama2:7b",
            url="http://0.0.0.0:11434/api/generate",
            generation_kwargs={
                "num_predict": 100,
                "temperature": 0.9,
            },
            timeout=300,
        )
        self.retriever_type = retriever_type
        self.retriever = self._get_retriever(retriever_type)
        self.wrapper = self._get_retriever_wrapper(self.retriever)


    def _get_retriever(self, retriever_type):
        retriever_factories = {
            "bm25": BM25RetrieverFactory(),
            "embeddings": EmbeddingRetrieverFactory(),
        }
        retriever_factory = retriever_factories.get(retriever_type)
        if retriever_factory:
            return retriever_factory.create_retriever(self.document_store)
        else:
            raise ValueError(f"Invalid retriever type: {retriever_type}")
        
    def _get_retriever_wrapper(self, retriever):
        retriever_wrappers = {
            "bm25": BM25RetrieverWrapper,
            "embeddings": EmbeddingRetrieverWrapper,
        }
        retriever_wrapper_cls = retriever_wrappers.get(self.retriever_type)
        if retriever_wrapper_cls:
            return retriever_wrapper_cls(retriever, self.llm)
        else:
            raise ValueError(f"Invalid retriever type: {self.retriever_type}")



    def completion(self, query, top_k=2):
        return self.wrapper.completion(query, top_k=top_k)
