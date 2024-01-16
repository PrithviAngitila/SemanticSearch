from retriever import BM25Retriever
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from ollama_haystack import OllamaGenerator
from haystack import Pipeline
from opensearch_haystack import OpenSearchBM25Retriever
from opensearch_haystack import OpenSearchDocumentStore




class Generator:
    def __init__(self):
        self.document_store = OpenSearchDocumentStore(
            hosts="http://localhost:9200", use_ssl=False, verify_certs=False, index='haystack'
        )
        self.retriever = OpenSearchBM25Retriever(document_store=self.document_store)
        self.generator = OllamaGenerator(model="llama2:7b",
                            url = "http://0.0.0.0:11434/api/generate",
                            generation_kwargs={
                              "num_predict": 100,
                              "temperature": 0.9,
                              },timeout=300)
        self.pipe = self._build_pipeline()

    def _get_template(self):
        prompt_template =   """
                            Given these documents, answer the question.\nDocuments:
                            {% for doc in documents %}
                                {{ doc.content }}
                            {% endfor %}

                            \nQuestion: {{query}}
                            \nAnswer:
                            """
        return prompt_template
    
    def _build_pipeline(self):
        pipe = Pipeline()
        pipe.add_component("retriever", self.retriever)
        pipe.add_component("prompt_builder", PromptBuilder(template=self._get_template()))
        pipe.add_component("llm", self.generator)
        pipe.add_component(instance=AnswerBuilder(), name="answer_builder")
        pipe.connect("retriever", "prompt_builder.documents")
        pipe.connect("prompt_builder", "llm") 
        pipe.connect("llm.replies", "answer_builder.replies")
        
        # print(self.generator.run("Who dicovered first antibiotic?"))
        return pipe

    def completion(self, query):
        result = self.pipe.run({"prompt_builder": {"query": query},
                                    "retriever": {"query": query, "top_k": 2}, "answer_builder": {"query": query}})
        print(result['answer_builder']['answers'])

        return result['answer_builder']['answers']

if __name__ == '__main__':
    gen = Generator()
    gen.completion(query="who won first olympics?")
        
