import streamlit as st
from dotenv import load_dotenv
import os

from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from pdf_processor import PDFProcessor
from text_processor import TextProcessor
from embedding_processor import EmbeddingProcessor
from qa_processor import QAProcessor
from logger import setup_logger

load_dotenv()

gemini_api = os.getenv('GEMINI_API')
logger = setup_logger()


class App:
    def __init__(self):
        st.title("Chemical Articles Q&A App")
        self.uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if self.uploaded_file:
            self.pdf_path = f"{self.uploaded_file.name}"
            with open(self.pdf_path, "wb") as pdf_file:
                pdf_file.write(self.uploaded_file.read())
            self.run_app()

    def extract_data(self, user_question):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api)
            vector_store = FAISS.load_local("faiss_index", embeddings)
            docs = vector_store.similarity_search(user_question)
            chain = QAProcessor("gemini-pro", gemini_api).get_conversational_chain()

            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

            st.write("Answer:", response["output_text"])
        except Exception as e:
            logger.error(f"Error in extract_data: {str(e)}")

    def run_app(self):
        try:
            raw_text = PDFProcessor(self.pdf_path).get_raw_text()
            text_chunks = TextProcessor(raw_text).get_text_chunks()

            st.info("PDF content successfully extracted!")

            user_question = st.text_input("Ask a question:")
            if st.button("Get Answer"):
                if user_question:
                    EmbeddingProcessor("models/embedding-001", gemini_api).get_vector_store(text_chunks)
                    self.extract_data(user_question)  # Use self to call instance method
                else:
                    st.warning("Please enter a question.")
        except Exception as e:
            logger.error(f"Error in run_app: {str(e)}")


if __name__ == "__main__":
    App()
