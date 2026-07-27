from app.rag.document_loader import PDFDocumentLoader
from app.rag.text_splitter import RecursiveTextSplitter
from app.schemas.document import PageContent


def test_chunking_produces_chunks_with_metadata(sample_pdf):
    loader = PDFDocumentLoader()
    splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=20)

    pages = loader.load(sample_pdf)
    chunks = splitter.split_pages(
        pages,
        document_id="doc-123",
        document_name="sample.pdf",
        source="sample.pdf",
    )

    assert len(chunks) > 1
    assert chunks[0].document_id == "doc-123"
    assert chunks[0].document_name == "sample.pdf"
    assert chunks[0].source == "sample.pdf"
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_id == "sample.pdf_chunk_0"
    assert len(chunks[0].content) <= 100


def test_chunking_respects_overlap():
    splitter = RecursiveTextSplitter(chunk_size=200, chunk_overlap=50)
    pages = [
        PageContent(page_number=1, text="Alpha content. " * 100),
        PageContent(page_number=2, text="Beta content. " * 100),
        PageContent(page_number=3, text="Gamma content. " * 100),
    ]

    chunks = splitter.split_pages(
        pages,
        document_id="doc-456",
        document_name="multi_page.pdf",
        source="multi_page.pdf",
    )

    assert len(chunks) > 3
    page_numbers = {chunk.page_number for chunk in chunks}
    assert page_numbers == {1, 2, 3}
    assert all(len(chunk.content) <= 200 for chunk in chunks)
