import fitz  # PyMuPDF
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_sahaja_yoga_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    
    # 1. Extract raw text from PDF
    for page in doc:
        full_text += page.get_text()

    # 2. Pattern to find the start of a lecture (e.g., 1970-01-01)
    # This matches the structure seen in your PDF
    lecture_pattern = r"(\d{4}-\d{2}-\d{2}),\s*(.*)\n"
    
    # Split the document into individual lectures
    lectures = re.split(lecture_pattern, full_text)
    
    # The first element is usually header junk before the first lecture
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    # Re-assemble lectures with their metadata
    # The split creates: [pre-text, date1, title1, content1, date2, title2, content2...]
    for i in range(1, len(lectures), 3):
        lecture_metadata = {
            "date": lectures[i],
            "title": lectures[i+1].strip(),
            "source": pdf_path
        }
        lecture_body = lectures[i+2]
        
        # Sub-chunking the specific lecture
        sub_chunks = text_splitter.create_documents(
            [lecture_body], 
            metadatas=[lecture_metadata]
        )
        all_chunks.extend(sub_chunks)

    return all_chunks

# Execute the processing
chunks = process_sahaja_yoga_pdf("Transcripts_English.pdf")

print(f"Total Chunks Created: {len(chunks)}")
print(f"Example Metadata: {chunks[0].metadata}")