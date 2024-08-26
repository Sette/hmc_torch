import os

import pytest

from app.api.classify.pipeline import classify_text
from app.config import settings
from app.main import app, custom_openapi
from app.utils.text import get_text

pdf = "./tests/docs/intimacao.pdf"
PWD = os.path.dirname(__file__)

@pytest.fixture
def get_pdf():
    def _loader(filename):
        with open(filename, "rb") as pdf_file:
            text = get_text(pdf_file.read(), req_id="10")
            return text

    return _loader


def test_classification(get_pdf):
    text_doc = get_pdf(pdf)
    results = classify_text(text_doc, "2123")
    assert results is not None


def test_health(client):
    url = f"{settings.base_path}/health"
    response = client.get(url)
    assert response.status_code == 200


def test_openapi_schema(client):
    response = custom_openapi()
    assert response is not None


def test_redoc(client):
    url = f"{settings.base_path}/redoc"
    response = client.get(url)
    assert response.status_code == 200


# EOF
