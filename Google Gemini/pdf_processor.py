import PyPDF2


class PDFProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def get_raw_text(self):
        with open(self.pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            raw_text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                raw_text += page.extract_text()
        return raw_text
