from .base_retriever import BaseRetrieverWrapper
from haystack import Pipeline
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder


class BM25RetrieverWrapper(BaseRetrieverWrapper):

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
        pipe.add_component("retriever", self.retriever)
        pipe.add_component(
            "prompt_builder", PromptBuilder(template=self.get_prompt())
        )
        pipe.add_component("llm", self.llm)
        pipe.add_component(instance=AnswerBuilder(), name="answer_builder")
        pipe.connect("retriever", "prompt_builder.documents")
        pipe.connect("prompt_builder", "llm")
        pipe.connect("llm.replies", "answer_builder.replies")
        return pipe

    def completion(self, query, top_k=2):
        result = self.pipeline.run(
            {
                "prompt_builder": {"query": query},
                "retriever": {"query": query, "top_k": top_k},
                "answer_builder": {"query": query},
            }
        )
        return result["answer_builder"]["answers"]
