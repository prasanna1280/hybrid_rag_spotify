from pathlib import Path
from pypdf import PdfReader


def extract_pdf(pdf_path: str):
    """Extract page-aware text from the supplied PDF."""
    reader = PdfReader(pdf_path)
    pages = []

    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        if text:
            pages.append({"page": page_no, "text": text})

    return pages


def section_chunk(pages, chunk_size=1200, overlap=180):
    """Create page-aware chunks while preserving enough local context."""
    chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": f"chunk_{chunk_id}",
                        "page": page["page"],
                        "text": chunk_text,
                    }
                )
                chunk_id += 1

            if end >= len(text):
                break

            start = max(0, end - overlap)

    return chunks
