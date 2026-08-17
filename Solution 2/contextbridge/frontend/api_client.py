"""HTTP client the Streamlit pages share.

Kept separate from the pages so error handling and the backend URL live in one
place — every page calls these functions rather than building requests itself.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
API = f"{BACKEND_URL}/api"

# Ingestion and extraction are slow by design (many LLM calls).
DEFAULT_TIMEOUT = 180.0
LONG_TIMEOUT = 900.0


class BackendError(RuntimeError):
    """A backend call failed — carries a message safe to show the user."""


def _call(
    method: str,
    path: str,
    timeout: float = DEFAULT_TIMEOUT,
    **kwargs: Any,
) -> Any:
    url = f"{API}{path}"
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.request(method, url, **kwargs)
    except httpx.ConnectError as exc:
        raise BackendError(
            f"Cannot reach the backend at {BACKEND_URL}. "
            "Start it with: uvicorn backend.main:app --port 8000"
        ) from exc
    except httpx.ReadTimeout as exc:
        raise BackendError(
            f"The backend took longer than {timeout:.0f}s to respond. "
            "Large documents can take a while on first ingest."
        ) from exc
    except httpx.HTTPError as exc:
        raise BackendError(f"Request failed: {exc}") from exc

    if response.status_code >= 400:
        detail = ""
        try:
            body = response.json()
            detail = body.get("detail") or body.get("error") or str(body)
        except Exception:
            detail = response.text[:500]
        raise BackendError(f"{response.status_code}: {detail}")

    return response.json()


# ----------------------------------------------------------------------
def health() -> dict[str, Any]:
    return _call("GET", "/health", timeout=15.0)


def upload_document(
    file_name: str, file_bytes: bytes, doc_type: str, run_summarization: bool
) -> dict[str, Any]:
    return _call(
        "POST",
        "/upload",
        timeout=LONG_TIMEOUT,
        files={"file": (file_name, file_bytes)},
        data={
            "doc_type": doc_type,
            "run_summarization": str(run_summarization).lower(),
        },
    )


def list_documents() -> dict[str, Any]:
    return _call("GET", "/documents", timeout=30.0)


def delete_document(doc_id: str) -> dict[str, Any]:
    return _call("DELETE", f"/documents/{doc_id}", timeout=60.0)


def chat(
    session_id: str,
    message: str,
    doc_id: str | None = None,
    mode: str = "auto",
    top_k: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "session_id": session_id,
        "message": message,
        "mode": mode,
    }
    if doc_id:
        payload["doc_id"] = doc_id
    if top_k:
        payload["top_k"] = top_k
    return _call("POST", "/chat", timeout=LONG_TIMEOUT, json=payload)


def get_session(session_id: str) -> dict[str, Any]:
    return _call("GET", f"/session/{session_id}", timeout=30.0)


def clear_session(session_id: str) -> dict[str, Any]:
    return _call("DELETE", f"/session/{session_id}", timeout=30.0)


def summarize(
    doc_id: str, level: str = "all", focus: str | None = None, refresh: bool = False
) -> dict[str, Any]:
    payload: dict[str, Any] = {"doc_id": doc_id, "level": level, "refresh": refresh}
    if focus:
        payload["focus"] = focus
    return _call("POST", "/summarize", timeout=LONG_TIMEOUT, json=payload)


def extract(
    doc_id: str,
    extraction_type: str = "all",
    client_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _call(
        "POST",
        "/extract",
        timeout=LONG_TIMEOUT,
        json={
            "doc_id": doc_id,
            "extraction_type": extraction_type,
            "client_profile": client_profile or {},
        },
    )


# ----------------------------------------------------------------------
@st.cache_data(ttl=10, show_spinner=False)
def cached_documents() -> list[dict[str, Any]]:
    """Document list, cached briefly so every page render isn't a round trip."""
    try:
        return list_documents().get("documents", [])
    except BackendError:
        return []


def document_selector(
    label: str = "Document", key: str = "doc_select", allow_none: bool = False
) -> str | None:
    """Shared document dropdown. Returns the selected doc_id."""
    documents = cached_documents()
    if not documents:
        st.info("No documents indexed yet — upload one on the **Upload** page.")
        return None

    options = {
        f"{d['file_name']}  ·  {d['chunk_count']} chunks  ·  {d['doc_type']}": d["doc_id"]
        for d in documents
    }
    if allow_none:
        options = {"(all documents)": None, **options}

    choice = st.selectbox(label, list(options), key=key)
    return options[choice]
