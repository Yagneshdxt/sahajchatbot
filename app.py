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
    # 3. Perform the search (k=100 for broader results)
    # Check if query changed to avoid re-running search on pagination
    if 'last_query' not in st.session_state or st.session_state.last_query != query:
        st.session_state.last_query = query
        st.session_state.search_results = db.similarity_search(query, k=100)
    
    results = st.session_state.search_results
    total_results = len(results)
    
    if total_results > 0:
        # Pagination Controls
        items_per_page = 5
        total_pages = (total_results + items_per_page - 1) // items_per_page
        
        col1, col2 = st.columns([1, 3])
        with col1:
            page = st.number_input("Page", min_value=1, max_value=total_pages, step=1, value=1)
        
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        current_batch = results[start_idx:end_idx]
        
        # Calculate displayed range for the header
        display_start = start_idx + 1
        display_end = min(start_idx + items_per_page, total_results)
        
        st.write(f"### Found {total_results} relevant sections (Showing results {display_start}-{display_end} of {total_results}):")
        
        for i, doc in enumerate(current_batch):
            # i is 0-indexed relative to batch, so add start_idx for global count
            match_num = start_idx + i + 1
            with st.expander(f"Match {match_num}: {doc.metadata.get('title', 'Unknown Title')}"):
                st.info(f"📅 **Date:** {doc.metadata.get('date')} | 📍 **Location:** {doc.metadata.get('location')}")
                st.write(doc.page_content)
                st.caption(f"Source: {doc.metadata.get('source')}")
    else:
        st.write("### No relevant sections found.")
