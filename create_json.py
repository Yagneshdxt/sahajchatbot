import fitz  # PyMuPDF
import re
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_to_json(pdf_path, output_json):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    # Regex based on document structure: YYYY-MM-DD followed by Title
    lecture_pattern = r"(\d{4}-\d{2}-\d{2}),\s*(.*)\n"
    parts = re.split(lecture_pattern, full_text)
    
    formatted_data = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    # Re-assemble matches: [junk, date, title, content, date, title, content...]
    for i in range(1, len(parts), 3):
        lecture_date = parts[i]
        lecture_title = parts[i+1].strip()
        lecture_body = parts[i+2]

        # Extract location if possible (usually found in the first few lines of body)
        location_match = re.search(r"\n(.*?)\s*\(India\)|\n(.*?)\s*\(England\)", lecture_body)
        location = location_match.group(0).strip() if location_match else "Unknown"

        # Split this specific lecture into chunks
        chunks = text_splitter.split_text(lecture_body)
        
        for chunk in chunks:
            formatted_data.append({
                "text": chunk,
                "metadata": {
                    "date": lecture_date,
                    "title": lecture_title,
                    "location": location,
                    "source": pdf_path
                }
            })

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=4)
    
    print(f"Extraction complete! Created {len(formatted_data)} chunks in {output_json}")

# Run Script 1
extract_to_json("Transcripts/Transcripts_English.pdf", "sahaja_data.json")
