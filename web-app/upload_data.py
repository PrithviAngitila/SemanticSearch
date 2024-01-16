import pandas as pd
import tqdm
from haystack import Document
from haystack.document_stores import DuplicatePolicy
from langchain_community.embeddings import OllamaEmbeddings
from ollama_haystack import OllamaGenerator
from opensearch_haystack import OpenSearchDocumentStore

embeddings = OllamaEmbeddings(model="llama2:7b", base_url= "http://0.0.0.0:11434", show_progress=True)
generator = OllamaGenerator(
    model="llama2:7b",
    url="http://0.0.0.0:11434/api/generate",
    generation_kwargs={
        "num_predict": 100,
        "temperature": 0.9,
    },
)
document_store = OpenSearchDocumentStore(
    hosts="http://localhost:9200",
    use_ssl=False,
    # verify_certs=False,
    http_auth=("admin", "admin"),
    index='haystack',
    embedding_dim=4096
)

# print(generator.run("Who is the best Indian actor?"))
# sample_vector = embeddings.embed_query("create an embedding")
# sample_doc = Document(content="sample content",meta={"url":"abc.com","title":"sample title"},embedding=sample_vector)
# document_store.write_documents(documents=[sample_doc], policy=DuplicatePolicy.SKIP)

df = pd.read_csv("data/dataset.csv")
documents = []

for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
    # vector = embeddings.embed_query(query)
    doc = Document(
        id=row["id"],
        content=row["text"],
        meta={"url": row["url"], "title": row["title"]},
        # embedding=vector,
    )
    # print(doc)
    documents.append(doc)

document_store.write_documents(documents=documents, policy=DuplicatePolicy.SKIP)
