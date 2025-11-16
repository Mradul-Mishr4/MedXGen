import os
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_file, text_split, download_embeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
print(">>> Script started running")

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medxgen-ayurvedic"
INDEX_DIM = 384   # MiniLM-L6-v2

# ---- Load and Process PDF ----
extracted_data = load_pdf_file(data="Data/")
text_chunks = text_split(extracted_data)
embeddings = download_embeddings()

# ---- Check/Create Index ----
existing_indexes = pc.list_indexes()

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=INDEX_DIM,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"Created new index: {index_name}")
else:
    print(f"Index {index_name} already exists")

# ---- Upload Chunks to Pinecone ----
print("Uploading documents to Pinecone...")

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name,
)

print("Upload completed!")
