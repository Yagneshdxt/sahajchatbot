import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# 1. Setup the UI
st.title("Sahaja Yoga Transcript Search")
query = st.text_input("Enter your search (e.g., Heart Chakra, Pune, Kundalini):")

# 2. Load the database (Make sure path matches your folder name)
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./sahaja_yoga_db", embedding_function=embeddings)

if query:
    # 3. Perform the search (k=5 means top 5 matches)
    results = db.similarity_search(query, k=5)
    
    st.write(f"### Found {len(results)} relevant sections:")
    
    for i, doc in enumerate(results):
        with st.expander(f"Match {i+1}: {doc.metadata.get('title', 'Unknown Title')}"):
            st.info(f"📅 **Date:** {doc.metadata.get('date')} | 📍 **Location:** {doc.metadata.get('location')}")
            st.write(doc.page_content)
            st.caption(f"Source: {doc.metadata.get('source')}")
