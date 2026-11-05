from pathlib import Path

from ingestion import parse_markdown_file


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
