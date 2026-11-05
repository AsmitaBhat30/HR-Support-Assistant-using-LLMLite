#!/usr/bin/env python3
"""
Ingestion Pipeline for SupportBot
Loads a single static markdown file into PostgreSQL vector database
"""

import asyncio
import sys
import os
from pathlib import Path
import json
import hashlib
from dataclasses import dataclass
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from litellm import embedding
from typing import List
import asyncpg

class Database:
    def __init__(self):
        self.pool = None

    async def connect(self):
        self.pool = await asyncpg.create_pool(dsn="postgresql://user:password@localhost:5432/rag_app")
        print("✅ Database connection pool established")

    async def disconnect(self):
        if self.pool:
            await self.pool.close()
            print("🛑 Database connection pool closed")

    async def get_pool(self):
        return self.pool

db = Database()

def get_embedding(text: str) -> List[float]:
    """
    Converts text into a vector using LiteLLM.
    """
    response = embedding(
        model="text-embedding-3-small",
        input=[text]
    )
    # The response format mimics OpenAI's standard API
    return response.data[0]["embedding"]


@dataclass
class DocumentChunk:
    page_content: str
    metadata: Dict[str, str]

    @property
    def chunk_id(self) -> str:
        raw = self.metadata["source"] + self.page_content
        return hashlib.md5(raw.encode("utf-8")).hexdigest()


def parse_markdown_file(file_path: Path) -> List[DocumentChunk]:
    text = file_path.read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=1000,
        chunk_overlap=100,
    )

    raw_chunks = splitter.create_documents([text])

    return [
        DocumentChunk(
            page_content=chunk.page_content,
            metadata={
                "source": file_path.name,
                "file_path": str(file_path),
                **chunk.metadata,
            },
        )
        for chunk in raw_chunks
    ]


async def ingest_chunks(chunks: List[DocumentChunk]) -> None:
    print("🧠 Generating embeddings...")
    embeddings = []
    for i, chunk in enumerate(chunks):
        if i % 10 == 0:
            print(f"  Processing chunk {i+1}/{len(chunks)}")
        embeddings.append(get_embedding(chunk.page_content))

    records = []
    for chunk, embedding in zip(chunks, embeddings):
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"
        records.append(
            (
                chunk.chunk_id,
                chunk.page_content,
                json.dumps(chunk.metadata),
                embedding_str,
                chunk.page_content,
            )
        )

    print(f"📥 Inserting {len(records)} records into database...")
    async with db.pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO document_chunks (id, content, metadata, embedding, text_search)
            VALUES ($1, $2, $3::jsonb, $4::vector, to_tsvector('english', $5))
            ON CONFLICT (id) DO NOTHING
            """,
            records,
        )

    print(f"✅ Successfully ingested {len(records)} chunks")


async def ingest_file(file_path: str) -> None:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"📄 Reading file: {path}")
    chunks = parse_markdown_file(path)
    print(f"📊 Created {len(chunks)} chunks")

    await ingest_chunks(chunks)


async def main() -> None:
    FILE_TO_INGEST = "Nimble_guide.MD"

    await db.connect()
    try:
        await ingest_file(FILE_TO_INGEST)
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())