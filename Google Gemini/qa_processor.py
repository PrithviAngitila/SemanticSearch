from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


class QAProcessor:
    def __init__(self, model, google_api_key):
        self.model = model
        self.google_api_key = google_api_key

    def get_conversational_chain(self):
        prompt_template = """
        Answer the question as crisp as possible from the provided context. If the necessary information
        is not present in the context, respond with 'Answer is not available in the context' rather
        than providing incorrect information.

        Context:\n{context}?\
        Question: \n{question}\n

        Answer:
        """

        model = ChatGoogleGenerativeAI(model=self.model, google_api_key=self.google_api_key, temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
