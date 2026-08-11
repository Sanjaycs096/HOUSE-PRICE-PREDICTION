import pytest
from app import create_app

@pytest.fixture
def app():
    app = create_app()
    app.config.update({
        "TESTING": True,
    })
    yield app

@pytest.fixture
def client(app):
    return app.test_client()

def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200

def test_predict_page(client):
    response = client.get("/predict")
    assert response.status_code == 200

def test_features_page(client):
    response = client.get("/features")
    assert response.status_code == 200
