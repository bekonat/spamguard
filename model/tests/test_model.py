import pytest
from fastapi.testclient import TestClient

from app import main
from app.text import clean_text, preprocess, stem
from app.train import train


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    report = train(out)
    assert report["test"]["f1"] > 0.9
    main.MODEL_PATH = out / "model.joblib"
    with TestClient(main.app) as c:
        yield c


def test_clean_text_replaces_entities():
    t = clean_text("Звоните +7 701 123 45 67 или http://a.top/x, всего 500 тенге")
    for tok in ("urltoken", "phonetoken", "numtoken"):
        assert tok in t


def test_homoglyphs_are_normalized():
    assert "бесплатно" in clean_text("беcплатно")


def test_stem_and_stopwords():
    assert stem("карты") == stem("карта")
    assert "и" not in preprocess("карта и деньги")
    assert preprocess("") == []


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["model_version"]


def test_predict_batch(client):
    r = client.post("/predict", json={"texts": [
        "Вы выиграли приз! Перейдите по ссылке http://win.top/x",
        "Привет, завтра лекция в 10:00, не опаздывай",
    ]})
    assert r.status_code == 200
    labels = [p["label"] for p in r.json()["predictions"]]
    assert labels == ["spam", "ham"]


@pytest.mark.parametrize("body", [{}, {"texts": []}, {"texts": ["x"] * 257}])
def test_predict_validation(client, body):
    assert client.post("/predict", json=body).status_code == 422


def test_metrics_exposed(client):
    client.post("/predict", json={"texts": ["привет"]})
    body = client.get("/metrics").text
    assert "model_predictions_total" in body
    assert "model_inference_seconds" in body
