from app.rag.document_loader import PDFDocumentLoader


def test_extract_text_from_pdf(sample_pdf):
    loader = PDFDocumentLoader()

    pages = loader.load(sample_pdf)

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "LocalizeRAG test content" in pages[0].text


def test_extract_text_from_multi_page_pdf(multi_page_pdf):
    loader = PDFDocumentLoader()

    pages = loader.load(multi_page_pdf)

    assert len(pages) == 3
    assert pages[0].page_number == 1
    assert pages[2].page_number == 3
    assert "Page 1 content" in pages[0].text
    assert "Page 3 content" in pages[2].text
