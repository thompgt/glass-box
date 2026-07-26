"""The HTTP surface: score a decision, then explain it.

Two routes carry the capability, and the interesting design is in how they fail.

``POST /predictions``
    Scores, explains, and durably spools before responding. A 200 here is a
    promise that the decision is on disk, not that it is in Iceberg yet.

``GET /explanations/{prediction_id}``
    Assembled from ``audit.*`` alone. The route has no access to the model, which
    is what makes the answer a record of that decision rather than a fresh
    computation against whatever model is loaded today.

Status codes are a claim about *what kind* of thing went wrong, so they are
mapped narrowly:

    422  the request named a feature the model does not have
    404  no such prediction id — an ordinary miss, usually a typo
    503  PROVENANCE_INTEGRITY_FAILURE — the id exists and the trail behind it is
         broken, or the loaded model no longer matches its recorded digest

Collapsing 404 and 503 would be the expensive mistake. A corrupt audit trail
reported as "not found" is a corrupt audit trail nobody investigates.

The spool is drained on shutdown, on a periodic tick, and on a read miss. That
last one is what makes a decision explainable the instant it is served: a lookup
that misses flushes once and retries before concluding the id does not exist.

``top_k`` is capped at 200, which is below the encoded width of some models — a
one-hot expansion of Adult's categoricals is already 107 columns and grows with
the vocabulary. That truncation is safe only because the residual is served
alongside the list: a caller that asks for 10 of 107 still receives a
decomposition that sums to the score. Were the residual dropped, the cap would
quietly turn a complete explanation into an incomplete one.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from ..catalog import glassbox_root, load_catalog
from ..train.registry import ProvenanceIntegrityError
from .explanations import DEFAULT_TOP_K, ExplanationNotFoundError, explanation_for
from .service import DEFAULT_THRESHOLD, PredictionService, UnknownFeatureError

INTEGRITY_FAILURE = "PROVENANCE_INTEGRITY_FAILURE"
FLUSH_INTERVAL_SECONDS = 5.0

# Which model version to serve, and at what threshold. Environment rather than a
# request parameter: a caller must not be able to choose the model that decides
# their case, and every prediction row must name a version the operator chose.
MODEL_VERSION_ENV = "GLASSBOX_MODEL_VERSION"
THRESHOLD_ENV = "GLASSBOX_THRESHOLD"


class PredictionRequest(BaseModel):
    """Raw features, exactly as ingest sees them.

    ``extra="allow"`` is deliberate: unknown fields reach the service, which
    rejects them by name. Rejecting them here as a schema violation would produce
    a less useful message, and silently discarding them would be worse than both.
    """

    model_config = ConfigDict(extra="allow")

    subject_id: str | None = Field(default=None, description="Optional data-subject identifier.")


def _service(request: Request) -> PredictionService:
    return request.app.state.service


# The one model version this process serves, resolved at startup. Injected rather
# than reached for, so a route cannot accidentally load a different one.
Served = Annotated[PredictionService, Depends(_service)]


def create_app(
    root: Path | None = None,
    model_version_id: str | None = None,
    *,
    threshold: float | None = None,
) -> FastAPI:
    """Build the app. Arguments override the environment, for tests and embedding."""
    resolved_root = Path(root) if root is not None else glassbox_root()
    resolved_version = model_version_id or os.environ.get(MODEL_VERSION_ENV)
    resolved_threshold = (
        threshold
        if threshold is not None
        else float(os.environ.get(THRESHOLD_ENV, DEFAULT_THRESHOLD))
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not resolved_version:
            raise RuntimeError(
                f"no model version to serve: pass model_version_id or set {MODEL_VERSION_ENV}"
            )

        catalog = load_catalog(resolved_root)
        # Fails closed at startup rather than per request. A process that cannot
        # verify its model against Iceberg should not accept traffic at all.
        app.state.service = PredictionService.load(
            catalog, resolved_version, resolved_root, threshold=resolved_threshold
        )
        app.state.flusher = asyncio.create_task(_flush_periodically(app.state.service))
        try:
            yield
        finally:
            app.state.flusher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await app.state.flusher
            # Last chance to commit anything served but not yet drained.
            await asyncio.to_thread(app.state.service.flush)

    app = FastAPI(
        title="Glass Box",
        description="A provenance-complete ML audit trail.",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(service: Served) -> dict[str, Any]:
        return {
            "status": "ok",
            "model_version_id": service.model_version_id,
            "artifact_digest": service.record["artifact_digest"],
            "recipe": service.record["recipe"],
            "threshold": service.threshold,
            "spool_pending": service.spool.pending_count(),
        }

    @app.post("/predictions", status_code=201)
    async def create_prediction(body: PredictionRequest, service: Served) -> dict[str, Any]:
        payload = body.model_dump(exclude_none=True)
        subject_id = payload.pop("subject_id", None)

        try:
            outcome = await asyncio.to_thread(service.predict, payload, subject_id=subject_id)
        except UnknownFeatureError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ProvenanceIntegrityError as exc:
            raise _integrity_failure(exc) from exc

        return {
            "prediction_id": outcome.prediction_id,
            "decision": outcome.decision,
            "score": outcome.score,
            "threshold": outcome.threshold,
            "model_version_id": outcome.model_version_id,
            # A pointer rather than the attributions themselves: the explanation
            # is an audit artifact assembled from the recorded trail, and serving
            # it from memory here would make it look like one when it is not.
            "explanation_url": f"/explanations/{outcome.prediction_id}",
        }

    @app.get("/explanations/{prediction_id}")
    async def get_explanation(
        prediction_id: str,
        service: Served,
        top_k: int = Query(default=DEFAULT_TOP_K, ge=1, le=200),
    ) -> dict[str, Any]:
        try:
            report = await asyncio.to_thread(
                _lookup, service, prediction_id, top_k, flush_on_miss=True
            )
        except ExplanationNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail=f"no recorded prediction {prediction_id!r}"
            ) from exc
        except ProvenanceIntegrityError as exc:
            raise _integrity_failure(exc) from exc

        return report.to_dict()

    return app


def _lookup(service: PredictionService, prediction_id: str, top_k: int, *, flush_on_miss: bool):
    """Look the decision up, draining the spool once if it is not committed yet.

    Without this a caller could be handed a prediction id and be told it does not
    exist a millisecond later, purely because the batch it belongs to has not been
    committed. The decision was durable the moment it was served; the wait for a
    batch is an implementation detail and must not surface as a 404.
    """
    try:
        return explanation_for(service.catalog, prediction_id, top_k=top_k)
    except ExplanationNotFoundError:
        if not flush_on_miss or service.spool.pending_count() == 0:
            raise
        service.flush()
        return explanation_for(service.catalog, prediction_id, top_k=top_k)


async def _flush_periodically(service: PredictionService) -> None:
    while True:
        await asyncio.sleep(FLUSH_INTERVAL_SECONDS)
        try:
            await asyncio.to_thread(service.flush)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            # A failed flush must not kill the drain loop: the spool is durable,
            # so the next tick retries, and the shutdown flush is the backstop.
            continue


def _integrity_failure(exc: Exception) -> HTTPException:
    """The refusal at the centre of this system, in HTTP form.

    503 rather than 500: the request was well formed and the system is the thing
    that is wrong. It will not serve a prediction it cannot attribute.
    """
    return HTTPException(
        status_code=503, detail={"error": INTEGRITY_FAILURE, "message": str(exc)}
    )
