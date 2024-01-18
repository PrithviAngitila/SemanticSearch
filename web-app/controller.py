from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack import Pipeline
from haystack_integrations.components.retrievers.opensearch import OpenSearchBM25Retriever, OpenSearchEmbeddingRetriever
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder


class Generator:
    def __init__(self, retriever_type="bm25"):
        self.document_store = OpenSearchDocumentStore(
            hosts="http://localhost:9200",
            use_ssl=False,
            verify_certs=False,
            index="wiki",
        )
        self.generator = OllamaGenerator(
            model="llama2:7b",
            url="http://0.0.0.0:11434/api/generate",
            generation_kwargs={
                "num_predict": 100,
                "temperature": 0.9,
            },
            timeout=300,
        )
        self.retriever_type=retriever_type 
        self.retriever = self._get_retriever(retriever_type)
        self.pipe = self._get_pipeline()

    def _get_retriever(self, retriever_type):
        if retriever_type == "bm25":
            return OpenSearchBM25Retriever(document_store=self.document_store)
        elif retriever_type == "embeddings":
            return OpenSearchEmbeddingRetriever(document_store=self.document_store)
        else:
            raise Exception(f'Invalid retriver type {retriever_type}') 

    def _get_pipeline(self):
        if self.retriever_type == "bm25":
            return self._getbm25_pipeline()
        elif self.retriever_type == "embeddings":
            return self._get_embedding_pipeline()
        else:
            raise Exception(f'Invalid retriver type {retriever_type}')  
    def _get_template(self):
        prompt_template = """
                            Given these documents, answer the question.\nDocuments:
                            {% for doc in documents %}
                                {{ doc.content }}
                            {% endfor %}

                            \nQuestion: {{query}}
                            \nAnswer:
                            """
        return prompt_template



    def _getbm25_pipeline(self):
        
        pipe = Pipeline()
        pipe.add_component("retriever", self.retriever)
        pipe.add_component(
            "prompt_builder", PromptBuilder(template=self._get_template())
        )
        pipe.add_component("llm", self.generator)
        pipe.add_component(instance=AnswerBuilder(), name="answer_builder")
        pipe.connect("retriever", "prompt_builder.documents")
        pipe.connect("prompt_builder", "llm")
        pipe.connect("llm.replies", "answer_builder.replies")
        return pipe
    
    def _get_embedding_pipeline(self):
        pipe = Pipeline()
        pipe.add_component("text_embedder", SentenceTransformersTextEmbedder(model='BAAI/bge-base-en-v1.5'))
        pipe.add_component("retriever", self.retriever)
        pipe.add_component(
            "prompt_builder", PromptBuilder(template=self._get_template())
        )
        pipe.add_component("llm", self.generator)
        pipe.add_component(instance=AnswerBuilder(), name="answer_builder")
        pipe.connect("text_embedder.embedding", "retriever.query_embedding")
        pipe.connect("retriever", "prompt_builder.documents")
        pipe.connect("prompt_builder", "llm")
        pipe.connect("llm.replies", "answer_builder.replies")
        return pipe
        

    def completion(self, query):
        if self.retriever_type == "bm25":
            result = self.pipe.run(
                {
                    "prompt_builder": {"query": query},
                    "retriever": {"query": query, "top_k": 2},
                    "answer_builder": {"query": query},
                }
            )
            return result["answer_builder"]["answers"]
        elif self.retriever_type == "embeddings":
            result = self.pipe.run({"text_embedder": {"text": query},"prompt_builder": {"query": query},
                    "answer_builder": {"query": query},})
            return result["answer_builder"]["answers"]

        raise Exception("Invalid retriver type")


if __name__ == "__main__":
    gen = Generator(retriever_type="embeddings")
    print(gen.completion(query="who won first olympics?"))
