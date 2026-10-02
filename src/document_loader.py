import pymupdf


def load_pdf(file_path):
    """
    Load text from a PDF document.
    """
    document = pymupdf.open(file_path)

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    return text
