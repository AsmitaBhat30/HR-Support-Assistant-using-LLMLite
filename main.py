from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

@dataclass
class DocumentChunk:
    page_content: str
    metadata: Dict[str, str]
    
    # We add a unique ID to ensure we don't ingest the same chunk twice (Idempotency)
    @property
    def chunk_id(self) -> str:
        # Create a deterministic hash of the source and content
        # We encode to utf-8 to ensure consistent hashing
        raw = self.metadata['source'] + self.page_content
        return hashlib.md5(raw.encode('utf-8')).hexdigest()



def parse_markdown_file(file_path: Path) -> List[DocumentChunk]:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Initialize the splitter with Markdown-specific rules
    # chunk_size=1000:  Target ~1000 characters per chunk (approx 200-300 tokens)
    # chunk_overlap=100: Keep 100 chars of context from the previous chunk
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=1000,
        chunk_overlap=100
    )
    
    # The splitter returns LangChain 'Document' objects
    raw_chunks = splitter.create_documents([text])
    
    # We convert them into our internal 'DocumentChunk' format
    processed_chunks = []
    for chunk in raw_chunks:
        processed_chunks.append(
            DocumentChunk(
                page_content=chunk.page_content,
                metadata={
                    "source": str(file_path.name),
                    "file_path": str(file_path),
                    # Inherit any existing metadata and add our own
                    **chunk.metadata
                }
            )
        )
        
    return processed_chunks

def main():
    """Main function to read a file and print chunks."""
    file_path = Path("/Users/suhas/Desktop/GitHub/HR-Support-Assistant-using-LLMLite/data/raw/hr-manual/platform-how-to-guides.md")
    
    print(f"📄 Reading file: {file_path}")
    
    chunks = parse_markdown_file(file_path)
    
    print(f"📊 Created {len(chunks)} chunks")
    print("=" * 80)
    
    # Print each chunk with metadata
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ---")
        print(f"Length: {len(chunk.page_content)} characters")
        print(f"Content:\n{chunk.page_content}")
        print("-" * 40)
    
    print(f"\n✅ Total chunks: {len(chunks)}")
    print(f"📄 File: {file_path}")
    print(f"📏 Total characters: {sum(len(chunk.page_content) for chunk in chunks)}")

if __name__ == "__main__":
    main()
