import pandas as pd
import tqdm
from haystack import Document
from haystack.document_stores.types.policy import DuplicatePolicy
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder

document_store = OpenSearchDocumentStore(
    hosts="http://localhost:9200",
    use_ssl=False,
    verify_certs=False,
    index='wiki',
)
df = pd.read_csv("./data/dataset.csv")
documents = []

for idx, row in tqdm.tqdm(df.head(1500).iterrows(), total=df.head(1500).shape[0]):
    # vector = embeddings.embed_query(query)
    doc = Document(
        id=row["id"],
        content=row["text"],
        meta={"url": row["url"], "title": row["title"]},
    )
    documents.append(doc)

model_name_or_path = "BAAI/bge-base-en-v1.5"
document_embedder = SentenceTransformersDocumentEmbedder(model=model_name_or_path, batch_size=64)  
document_embedder.warm_up()
documents_with_embeddings = document_embedder.run(documents)

document_store.write_documents(documents_with_embeddings.get("documents"),policy=DuplicatePolicy.SKIP)
