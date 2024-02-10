from .base_retriever import BaseRetrieverWrapper
from haystack import Pipeline
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder


class EmbeddingRetrieverWrapper(BaseRetrieverWrapper):

    def __init__(self, retriever, llm):
        super().__init__(retriever, llm)
        self.pipeline = self.get_pipeline() 

    def get_prompt(self):
        return """
                            Given these documents, answer the question.\nDocuments:
                            {% for doc in documents %}
                                {{ doc.content }}
                            {% endfor %}

                            \nQuestion: {{query}}
                            \nAnswer:
                            """

    def get_pipeline(self):
        pipe = Pipeline()
        pipe.add_component("text_embedder", SentenceTransformersTextEmbedder(model='BAAI/bge-base-en-v1.5'))
        pipe.add_component("retriever", self.retriever)
        pipe.add_component(
            "prompt_builder", PromptBuilder(template=self.get_prompt())
        )
        pipe.add_component("llm", self.llm)
        pipe.add_component(instance=AnswerBuilder(), name="answer_builder")
        pipe.connect("text_embedder.embedding", "retriever.query_embedding")
        pipe.connect("retriever", "prompt_builder.documents")
        pipe.connect("prompt_builder", "llm")
        pipe.connect("llm.replies", "answer_builder.replies")
        return pipe

    def completion(self, query, top_k=2):
        result = self.pipeline.run(
            {
                "text_embedder": {"text": query},
                "prompt_builder": {"query": query},
                "answer_builder": {"query": query},
            }
        )
        return result["answer_builder"]["answers"]