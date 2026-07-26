"""The HTTP surface, and specifically what it does when things are wrong.

The happy path is one test. The rest are about status codes, because on this
service a status code is a claim about what kind of failure occurred, and the
expensive mistake is reporting a broken audit trail as an ordinary miss.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from glassbox.bootstrap import bootstrap
from glassbox.catalog import load_catalog
from glassbox.ingest import ingest_adult
from glassbox.schemas import ATTRIBUTIONS, MODEL_VERSIONS
from glassbox.serve.app import INTEGRITY_FAILURE, create_app
from glassbox.train import train_baseline
from tests.test_serving import APPLICANT

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def deployment(tmp_path_factory):
    """One trained model, served over HTTP, shared across the module.

    Module-scoped because training and loading it costs far more than every test
    below combined. The two tests that corrupt the trail get their own
    deployment, since a shared one would leak the corruption forward.
    """
    root = tmp_path_factory.mktemp("api")
    bootstrap(root)
    ingest_adult(root)
    trained = train_baseline(root=root)

    app = create_app(root=root, model_version_id=trained.model_version.model_version_id)
    with TestClient(app) as client:
        yield client, root, trained.model_version.model_version_id


@pytest.fixture
def client(deployment):
    return deployment[0]


def predict(client, **overrides) -> dict:
    response = client.post("/predictions", json={**APPLICANT, **overrides})
    assert response.status_code == 201, response.text
    return response.json()


# ------------------------------------------------------------ happy path ----

def test_a_decision_can_be_explained_the_instant_it_is_served(client):
    """No wait for a batch. The spool drains on a read miss so a 404 cannot mean
    'committed soon'."""
    created = predict(client, subject_id="subject-1")

    response = client.get(created["explanation_url"])
    assert response.status_code == 200, response.text
    report = response.json()

    assert report["prediction_id"] == created["prediction_id"]
    assert report["decision"] == created["decision"]
    assert report["score"] == pytest.approx(created["score"])
    assert report["subject_id"] == "subject-1"
    assert report["explanation"]["reconciles"] is True
    assert report["model"]["artifact_digest"]
    assert report["training_data"]["row_count"] > 0


def test_the_prediction_response_points_at_the_audit_record_rather_than_inlining_it(client):
    created = predict(client)

    assert created["explanation_url"] == f"/explanations/{created['prediction_id']}"
    assert "attributions" not in created


def test_top_k_truncates_the_served_list_and_the_residual_absorbs_the_rest(client):
    created = predict(client)

    full = client.get(created["explanation_url"], params={"top_k": 200}).json()
    short = client.get(created["explanation_url"], params={"top_k": 3}).json()

    assert len(short["explanation"]["attributions"]) == 3
    assert len(full["explanation"]["attributions"]) == full["explanation"]["total_feature_count"]
    # Nothing was truncated, so there is no tail for the residual to carry.
    assert full["explanation"]["residual"] == 0.0
    assert abs(short["explanation"]["residual"]) > 0.0
    assert short["explanation"]["reconciles"] is True
    assert full["explanation"]["reconciles"] is True


def test_health_reports_the_model_it_is_actually_serving(client, deployment):
    _, _, model_version_id = deployment

    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["model_version_id"] == model_version_id
    assert body["artifact_digest"]


# --------------------------------------------------------- status codes ----

def test_an_unknown_feature_is_a_422(client):
    response = client.post("/predictions", json={**APPLICANT, "favourite_colour": "blue"})

    assert response.status_code == 422
    assert "favourite_colour" in response.json()["detail"]


def test_an_unknown_prediction_id_is_a_404(client):
    response = client.get("/explanations/not-a-real-prediction")

    assert response.status_code == 404
    assert INTEGRITY_FAILURE not in response.text


def test_a_broken_trail_is_a_503_and_names_itself(tmp_path_factory):
    """The refusal at the centre of the system: it will not serve a prediction it
    cannot attribute, and it says which failure this is."""
    root = tmp_path_factory.mktemp("api-broken")
    bootstrap(root)
    ingest_adult(root)
    trained = train_baseline(root=root)

    app = create_app(root=root, model_version_id=trained.model_version.model_version_id)
    with TestClient(app) as client:
        created = predict(client)
        client.get(created["explanation_url"])  # force the spool to drain

        from pyiceberg.expressions import EqualTo

        catalog = load_catalog(root)
        catalog.load_table(ATTRIBUTIONS.identifier).delete(
            EqualTo("prediction_id", created["prediction_id"])
        )

        response = client.get(created["explanation_url"])

    assert response.status_code == 503
    assert response.json()["detail"]["error"] == INTEGRITY_FAILURE


def test_a_model_version_with_no_provenance_record_never_accepts_traffic(tmp_path_factory):
    """Fails at startup, not per request. A process that cannot verify its model
    against Iceberg should not be serving at all."""
    from glassbox.train.registry import ProvenanceIntegrityError

    root = tmp_path_factory.mktemp("api-unregistered")
    bootstrap(root)

    app = create_app(root=root, model_version_id="not-a-real-model-version")
    with pytest.raises(ProvenanceIntegrityError):
        with TestClient(app):
            pass


def test_serving_without_a_configured_model_version_refuses_to_start(tmp_path_factory, monkeypatch):
    monkeypatch.delenv("GLASSBOX_MODEL_VERSION", raising=False)
    root = tmp_path_factory.mktemp("api-unconfigured")
    bootstrap(root)

    with pytest.raises(RuntimeError, match="no model version to serve"):
        with TestClient(create_app(root=root)):
            pass


# ------------------------------------------------------------- recording ----

def test_every_served_decision_reaches_the_audit_tables(client, deployment):
    _, root, model_version_id = deployment
    ids = [predict(client)["prediction_id"] for _ in range(3)]

    # The last read forces a drain of anything still spooled.
    client.get(f"/explanations/{ids[-1]}")

    catalog = load_catalog(root)
    for prediction_id in ids:
        assert client.get(f"/explanations/{prediction_id}").status_code == 200

    model_rows = catalog.load_table(MODEL_VERSIONS.identifier).scan().to_arrow().to_pylist()
    assert model_version_id in {r["model_version_id"] for r in model_rows}
