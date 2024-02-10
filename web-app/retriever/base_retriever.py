
class BaseRetrieverWrapper:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def get_prompt(self, query):
        raise NotImplementedError("Subclasses must implement get_prompt method")

    def get_pipeline(self):
        raise NotImplementedError("Subclasses must implement get_pipeline method")

    def completion(self, query, top_k=2):
        raise NotImplementedError("Subclasses must implement completion method")
