from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextProcessor:
    def __init__(self, text):
        self.text = text

    def get_text_chunks(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(self.text)
        return chunks
