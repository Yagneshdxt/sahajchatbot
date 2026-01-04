import json
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings

def upload_json_to_chroma(json_file, db_folder):
    # 1. Load your verified data
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. Convert JSON entries back to LangChain Documents
    documents = [
        Document(page_content=item["text"], metadata=item["metadata"]) 
        for item in data
    ]

    # 3. Use a free local embedding model (runs on your CPU)
    # This model is small, fast, and great for learning.
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Create and persist the database
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=db_folder
    )
    
    print(f"Successfully uploaded {len(documents)} chunks to {db_folder}")

# Run Script 2
upload_json_to_chroma("sahaja_data.json", "./sahaja_yoga_db")
