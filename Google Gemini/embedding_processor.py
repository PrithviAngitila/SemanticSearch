from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS


class EmbeddingProcessor:
    def __init__(self, model, google_api_key):
        self.model = model
        self.google_api_key = google_api_key

    def get_vector_store(self, text_chunks):
        embeddings = GoogleGenerativeAIEmbeddings(model=self.model, google_api_key=self.google_api_key)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
