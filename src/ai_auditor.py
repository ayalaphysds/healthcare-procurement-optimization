"""
Phase 2 GenAI, Regulatory RAG, Security Guardrails & Human-in-the-Loop Auditor.

Bridges the Phase 1 Quantitative Optimization Engine (`ProcurementOptimizer`) with
the production GenAI engineering patterns developed in the AI Specialization (Gate 1 & Gate 2):

1. Resilient 3-Layer LLM Client (`ResilientGeminiClient`):
   - Separates non-retryable client/auth errors (HTTP 400/401/403 -> `ValueError` fail-fast)
     from transient rate-limit/network failures (HTTP 429/500/503 -> `TimeoutError` -> retry
     with configurable backoff -> `RuntimeError` on exhaustion).
   - Computes real-time token costs from `usageMetadata` and provides a deterministic
     high-fidelity offline synthesizer when `GEMINI_API_KEY` is not set.
2. Prompt-Injection & Document Poisoning Guardrails (`validate_query`, `filter_suspicious_chunks`):
   - Blocks adversarial jailbreak/override instructions at both query ingress and chunk retrieval.
3. Regulatory RAG Engine with Mandatory Source Citations (`RegulatoryRAGService`):
   - Indexes CENABAST & Chilean Public Procurement (`Law 19,886`) normative documents into
     semantic vector chunks, retrieves top-k clauses, and enforces mandatory inline citations
     (`[source: <doc_id>]`).
4. Structured JSON Downstream Interface Contract (`audit_tender_bid`):
   - Emits a strict JSON Schema Draft 2020-12 compliant payload (`overprice_detected`,
     `deviation_pct`, `recommendation`, `ruling`, `citations`) for downstream microservices.
5. PII-Free Structured JSON Observability (`trace_id_var`, `log_telemetry_event`):
   - Propagates a per-request `trace_id` via Python `contextvars` and logs operational metadata
     (character lengths, token counts, latency, retrieval k) with zero PII or raw prompt dumps.
6. Human-in-the-Loop (HITL) Stateful Supervision Gate (`SupervisionWorkflow`):
   - Pauses high-risk or overpriced tender rulings at an explicit `interrupt` checkpoint and
     resumes only upon human supervisor approval or rejection (`resume(approved=...)`).
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# 1. OBSERVABILITY LAYER (CONTEXTVARS TRACE_ID & ZERO-PII STRUCTURED JSON LOGS)
# -----------------------------------------------------------------------------
trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "trace_id", default="no-trace"
)

# Forbidden raw-content / PII keys that must never appear in telemetry logs
FORBIDDEN_LOG_KEYS = frozenset(
    {"prompt", "raw_text", "question", "pregunta", "email", "rut", "tax_id", "api_key"}
)


def new_trace_id() -> str:
    """Generate a correlation trace ID for end-to-end request observability."""
    return f"trc-{uuid.uuid4().hex[:12]}"


def estimate_tokens(text: str) -> int:
    """Fast heuristic token estimator (~4 characters per token, minimum 1)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def log_telemetry_event(event_name: str, **metadata: Any) -> Dict[str, Any]:
    """
    Create a structured JSON observability event correlated by `trace_id`,
    strictly stripping any raw text or PII keys.
    """
    safe_meta = {
        k: v for k, v in metadata.items() if k.lower() not in FORBIDDEN_LOG_KEYS
    }
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trace_id": trace_id_var.get(),
        "event": event_name,
        **safe_meta,
    }


# -----------------------------------------------------------------------------
# 2. SECURITY LAYER (PROMPT-INJECTION & CHUNK-POISONING GUARDRAILS)
# -----------------------------------------------------------------------------
INJECTION_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"ignora\s+(todas\s+)?las\s+instrucciones", re.IGNORECASE),
    re.compile(r"system\s*prompt|developer\s*mode|jailbreak", re.IGNORECASE),
    re.compile(r"olvida\s+tu\s+rol|forget\s+your\s+role", re.IGNORECASE),
    re.compile(r"reveal\s+(secret|api[_\s]?key|credentials)", re.IGNORECASE),
    re.compile(r"override\s+procurement\s+cap|bypass\s+audit", re.IGNORECASE),
)

OUT_OF_CONTEXT_FALLBACK = (
    "Insufficient verified regulatory context to answer this query safely. "
    "Please restrict questions to CENABAST pharmaceutical procurement regulations, "
    "fair-price bands, and supplier SLA rules."
)


def validate_query(query: str) -> Tuple[bool, str]:
    """
    Screen incoming user query for empty input or adversarial prompt injection.
    Returns `(is_valid, reason)`.
    """
    if not query or not query.strip():
        return False, "empty_query"
    cleaned = query.strip()
    if len(cleaned) > 2000:
        return False, "query_exceeds_max_length"
    for pattern in INJECTION_PATTERNS:
        if pattern.search(cleaned):
            return False, "prompt_injection_detected"
    return True, "ok"


def filter_suspicious_chunks(
    chunks: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Inspect retrieved document chunks and quarantine any chunk containing
    indirect prompt-injection payloads before building the LLM context window.
    """
    clean_chunks: List[Dict[str, Any]] = []
    discarded_chunks: List[Dict[str, Any]] = []
    for chunk in chunks:
        text = str(chunk.get("content", ""))
        if any(p.search(text) for p in INJECTION_PATTERNS):
            discarded_chunks.append(chunk)
        else:
            clean_chunks.append(chunk)
    return clean_chunks, discarded_chunks


# -----------------------------------------------------------------------------
# 3. RESILIENT 3-LAYER LLM CLIENT (TRANSPORT -> RETRY -> UNIFIED INTERFACE)
# -----------------------------------------------------------------------------
# Gemini 2.5 Flash reference pricing per 1M tokens (USD)
GEMINI_FLASH_INPUT_USD_PER_1M = 0.15
GEMINI_FLASH_OUTPUT_USD_PER_1M = 0.60


def calculate_llm_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Calculate exact request cost in USD from token usage metadata."""
    in_cost = (max(0, input_tokens) / 1_000_000.0) * GEMINI_FLASH_INPUT_USD_PER_1M
    out_cost = (max(0, output_tokens) / 1_000_000.0) * GEMINI_FLASH_OUTPUT_USD_PER_1M
    return round(in_cost + out_cost, 8)


def with_retry(
    operation: Callable[[], Any],
    max_attempts: int = 3,
    backoff_seconds: float = 0.05,
) -> Any:
    """
    Higher-order retry wrapper (`cliente_reintento` pattern from Gate 2):
    - Propagates `ValueError` (400/401/403 authentication or schema errors) immediately.
    - Retries `TimeoutError` (429 rate limit / 500 / 503 / transient network failure)
      up to `max_attempts` with exponential backoff.
    - Raises `RuntimeError` if all attempts are exhausted.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1.")

    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except ValueError:
            # Fail-fast on non-retryable client/auth errors
            raise
        except TimeoutError as exc:
            last_err = exc
            if attempt < max_attempts and backoff_seconds > 0:
                time.sleep(backoff_seconds * (2 ** (attempt - 1)))

    raise RuntimeError(
        f"LLM API request failed after {max_attempts} attempts: {last_err}"
    ) from last_err


def classify_http_status(status_code: int, detail: str = "") -> None:
    """
    Classify HTTP response codes into fail-fast `ValueError` vs retryable `TimeoutError`.
    """
    if status_code == 200:
        return
    if status_code in (400, 401, 403):
        raise ValueError(f"Non-retryable client/auth error (HTTP {status_code}): {detail}")
    if status_code in (408, 429, 500, 502, 503, 504):
        raise TimeoutError(f"Transient API/rate-limit error (HTTP {status_code}): {detail}")
    raise RuntimeError(f"Unexpected HTTP status {status_code}: {detail}")


# -----------------------------------------------------------------------------
# 4. NORMATIVE CORPUS & VECTOR RAG ENGINE WITH MANDATORY CITATIONS
# -----------------------------------------------------------------------------
CENABAST_REGULATORY_CORPUS: Tuple[Dict[str, str], ...] = (
    {
        "doc_id": "ley_19886_compras_publicas.md",
        "title": "Chilean Public Procurement Law No. 19,886 — Transparency & Value for Money",
        "content": (
            "Article 6 & 9: Public healthcare tenders managed by CENABAST must maximize fiscal "
            "efficiency while ensuring continuous clinical supply. Any supplier bid exceeding "
            "the empirical 75th percentile (P75 fair-price ceiling) for an identical active "
            "ingredient requires mandatory secondary economic justification or price negotiation "
            "before contract adjudication."
        ),
    },
    {
        "doc_id": "directiva_cenabast_multioferente_2026.md",
        "title": "CENABAST Multi-Supplier Split-Award Directive (Anti-Monopoly Cap)",
        "content": (
            "Section 4.2: To prevent public hospital drug stockouts and reduce Herfindahl-Hirschman "
            "Index (HHI) concentration below 4,500 points per molecule, critical formulary tenders "
            "must enforce a maximum single-supplier allocation cap of 45% (alpha <= 0.45) of annual "
            "national demand, guaranteeing at least three qualified laboratories per essential drug."
        ),
    },
    {
        "doc_id": "manual_sla_logistica_hospitalaria.md",
        "title": "Hospital Delivery SLA & Cold-Chain Reliability Standard",
        "content": (
            "Section 2.1: Suppliers bidding on CENABAST tenders must maintain a minimum historical "
            "composite SLA reliability score of 70.0 / 100 and mean delivery delay <= 5 business days. "
            "Biologics, insulins, and vaccines require validated cold-chain transport (2°C to 8°C)."
        ),
    },
    {
        "doc_id": "catalogo_bandas_precios_cenabast_2026.md",
        "title": "CENABAST 2026 Reference Price Observatory & Overpricing Thresholds",
        "content": (
            "Section 1.3: Empirical interquartile reference bands (P25 to P75) benchmark fair unit "
            "prices across oral generics (Acetaminophen 500mg, Losartan 50mg, Metformin 850mg, "
            "Atorvastatin 20mg, Amoxicillin 500mg), high-cost oncology biologics (Pembrolizumab, "
            "Trastuzumab), and surgical supplies. Deviations > 15% above median trigger negotiation; "
            "deviations > 30% above median trigger rejection or Human-in-the-Loop supervisor review."
        ),
    },
    {
        "doc_id": "protocolo_supervision_humana_hitl.md",
        "title": "Governance Protocol for Automated AI Procurement Auditing (HITL)",
        "content": (
            "Section 3.0: Automated AI tender assessments that detect severe overpricing (>15% "
            "above benchmark median) or recommend contract rejection must pause execution via an "
            "interactive supervision checkpoint (interrupt) and await explicit human procurement "
            "officer sign-off before updating downstream ERP award schedules."
        ),
    },
)


@dataclass
class RegulatoryVectorIndex:
    """
    Lightweight deterministic vector index over CENABAST regulatory documents.
    Uses normalized character/word n-gram TF-IDF embeddings and cosine similarity,
    mirroring the ChromaDB persistent vectorstore behavior from Gate 2 without
    requiring external embedding API keys during CI/CD tests.
    """

    chunks: List[Dict[str, str]] = field(default_factory=list)
    vocab: Dict[str, int] = field(default_factory=dict)
    idf: Optional[np.ndarray] = None
    matrix: Optional[np.ndarray] = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        words = re.findall(r"[a-z0-9_]+", text.lower())
        tokens = list(words)
        # Add character 4-grams for cross-lingual & morphological resilience
        compact = "".join(words)
        tokens.extend(compact[i : i + 4] for i in range(max(0, len(compact) - 3)))
        return tokens

    def build(self, corpus: Tuple[Dict[str, str], ...] = CENABAST_REGULATORY_CORPUS) -> None:
        self.chunks = [dict(doc) for doc in corpus]
        tokenized_docs = [
            self._tokenize(f"{doc['title']} {doc['content']}") for doc in self.chunks
        ]

        vocab_set = sorted({tok for doc_toks in tokenized_docs for tok in doc_toks})
        self.vocab = {tok: idx for idx, tok in enumerate(vocab_set)}

        n_docs = len(self.chunks)
        n_vocab = len(self.vocab)
        tf = np.zeros((n_docs, n_vocab), dtype=float)
        df = np.zeros(n_vocab, dtype=float)

        for i, toks in enumerate(tokenized_docs):
            seen = set()
            for t in toks:
                j = self.vocab[t]
                tf[i, j] += 1.0
                seen.add(j)
            for j in seen:
                df[j] += 1.0

        self.idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        tfidf = tf * self.idf
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = tfidf / norms

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        if self.matrix is None or self.idf is None:
            self.build()
        assert self.matrix is not None and self.idf is not None

        q_toks = self._tokenize(query)
        q_vec = np.zeros(len(self.vocab), dtype=float)
        for t in q_toks:
            if t in self.vocab:
                q_vec[self.vocab[t]] += 1.0
        q_vec = q_vec * self.idf
        q_norm = float(np.linalg.norm(q_vec))
        if q_norm > 0:
            q_vec = q_vec / q_norm

        scores = self.matrix @ q_vec
        top_indices = np.argsort(scores)[::-1][: max(1, k)]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            doc = self.chunks[int(idx)]
            results.append(
                {
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "content": doc["content"],
                    "similarity_score": round(float(scores[int(idx)]), 4),
                }
            )
        return results


# -----------------------------------------------------------------------------
# 5. GATE 1 DOWNSTREAM JSON INTERFACE CONTRACT & AUDIT ENGINE
# -----------------------------------------------------------------------------
TENDER_AUDIT_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "CenabastTenderAuditContract",
    "type": "object",
    "properties": {
        "overprice_detected": {"type": "boolean"},
        "deviation_pct": {"type": "number", "minimum": -100.0},
        "recommendation": {"type": "string", "enum": ["award", "negotiate", "reject"]},
        "ruling": {"type": "string", "minLength": 10, "maxLength": 600},
        "citations": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    },
    "required": [
        "overprice_detected",
        "deviation_pct",
        "recommendation",
        "ruling",
        "citations",
    ],
}


@dataclass(frozen=True)
class TenderAuditResult:
    """Typed representation of the Gate 1 + Gate 2 downstream JSON contract."""

    trace_id: str
    active_ingredient: str
    offered_unit_price: float
    benchmark_median_price: float
    fair_ceiling_p75_price: float
    currency: str
    overprice_detected: bool
    deviation_pct: float
    recommendation: str
    ruling: str
    citations: List[str]
    requires_human_supervision: bool
    telemetry_events: List[Dict[str, Any]]

    def to_contract_dict(self) -> Dict[str, Any]:
        """Return the strict downstream JSON interface payload."""
        return {
            "overprice_detected": self.overprice_detected,
            "deviation_pct": self.deviation_pct,
            "recommendation": self.recommendation,
            "ruling": self.ruling,
            "citations": self.citations,
        }

    def to_full_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# 6. HUMAN-IN-THE-LOOP (HITL) STATEFUL SUPERVISION WORKFLOW
# -----------------------------------------------------------------------------
@dataclass
class SupervisionState:
    """
    Represents a paused or completed Human-in-the-Loop (HITL) procurement state,
    implementing the `interrupt` / `Command(resume=...)` pattern from Gate 2.
    """

    thread_id: str
    status: str  # "PAUSED_FOR_HUMAN_SUPERVISION" | "APPROVED" | "REJECTED" | "AUTO_CLEARED"
    audit_result: TenderAuditResult
    interrupt_payload: Optional[Dict[str, Any]] = None
    supervisor_decision: Optional[str] = None
    supervisor_notes: str = ""
    final_action: str = ""


class SupervisionWorkflow:
    """
    Stateful Human-in-the-Loop checkpoint manager for high-risk procurement awards.
    """

    def __init__(self) -> None:
        self._threads: Dict[str, SupervisionState] = {}

    def invoke(self, audit_result: TenderAuditResult, thread_id: str = "hitl-thread-1") -> SupervisionState:
        """
        Evaluate whether the audit ruling requires human supervision. If `requires_human_supervision`
        is True, pause execution at an `interrupt` checkpoint; otherwise auto-clear.
        """
        if audit_result.requires_human_supervision:
            interrupt_info = {
                "reason": "Tender bid exceeds P75 fair-price ceiling or 15% median deviation threshold.",
                "active_ingredient": audit_result.active_ingredient,
                "deviation_pct": audit_result.deviation_pct,
                "proposed_recommendation": audit_result.recommendation,
                "required_action": "Supervisor must approve override or confirm rejection/negotiation.",
            }
            state = SupervisionState(
                thread_id=thread_id,
                status="PAUSED_FOR_HUMAN_SUPERVISION",
                audit_result=audit_result,
                interrupt_payload=interrupt_info,
                final_action="PENDING_SUPERVISOR_REVIEW",
            )
        else:
            state = SupervisionState(
                thread_id=thread_id,
                status="AUTO_CLEARED",
                audit_result=audit_result,
                interrupt_payload=None,
                supervisor_decision="auto_approved",
                final_action=f"EXECUTE_{audit_result.recommendation.upper()}",
            )
        self._threads[thread_id] = state
        return state

    def resume(
        self,
        thread_id: str,
        approved: bool,
        supervisor_notes: str = "",
    ) -> SupervisionState:
        """
        Resume a paused workflow thread after human supervisor review (`Command(resume=...)`).
        """
        if thread_id not in self._threads:
            raise KeyError(f"Unknown supervision thread_id: {thread_id}")

        state = self._threads[thread_id]
        state.supervisor_notes = supervisor_notes.strip()
        if approved:
            state.status = "APPROVED"
            state.supervisor_decision = "approved_by_human_supervisor"
            state.final_action = f"CONFIRMED_{state.audit_result.recommendation.upper()}"
        else:
            state.status = "REJECTED"
            state.supervisor_decision = "overridden_to_reject_by_supervisor"
            state.final_action = "BLOCKED_BY_SUPERVISOR"
        return state


# -----------------------------------------------------------------------------
# 7. UNIFIED REGULATORY RAG & TENDER AUDITOR SERVICE
# -----------------------------------------------------------------------------
class RegulatoryRAGService:
    """
    Production RAG + Structured Tender Auditor combining:
    - Security validation (`validate_query`, `filter_suspicious_chunks`)
    - Vector retrieval over CENABAST procurement regulations (`RegulatoryVectorIndex`)
    - Mandatory source citation enforcement (`[source: <doc_id>]`)
    - Zero-PII structured JSON telemetry (`trace_id_var`, `log_telemetry_event`)
    - Optional live Gemini REST API call with 3-layer retry or deterministic offline synthesis
    """

    def __init__(self, vector_index: Optional[RegulatoryVectorIndex] = None) -> None:
        self.index = vector_index or RegulatoryVectorIndex()
        self.index.build()

    @staticmethod
    def compute_prompt_sha256(prompt_template: str) -> str:
        """Compute deterministic SHA-256 governance hash for prompt version registry."""
        return hashlib.sha256(prompt_template.encode("utf-8")).hexdigest()

    def query_regulations(
        self,
        question: str,
        top_k: int = 3,
        use_live_api: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute the full Gate 2 RAG lifecycle:
        1. Set `trace_id` context variable
        2. Screen query for prompt injection (`validate_query`)
        3. Retrieve top-k regulatory chunks (`RegulatoryVectorIndex.retrieve`)
        4. Filter retrieved chunks against indirect injection (`filter_suspicious_chunks`)
        5. Generate grounded response with mandatory `[source: <doc_id>]` citations
        6. Emit PII-free structured telemetry events
        """
        token = trace_id_var.set(new_trace_id())
        tid = trace_id_var.get()
        t0 = time.perf_counter()
        telemetry: List[Dict[str, Any]] = []

        def emit(event: str, **fields: Any) -> None:
            telemetry.append(log_telemetry_event(event, **fields))

        try:
            emit("query_received", query_char_len=len(question or ""), top_k=top_k)

            is_valid, sec_reason = validate_query(question)
            if not is_valid:
                emit("query_blocked", reason=sec_reason)
                return {
                    "trace_id": tid,
                    "answer": OUT_OF_CONTEXT_FALLBACK,
                    "sources": [],
                    "telemetry": telemetry,
                }

            retrieved = self.index.retrieve(question, k=top_k)
            emit("documents_retrieved", count=len(retrieved))

            clean_chunks, discarded = filter_suspicious_chunks(retrieved)
            if discarded:
                emit("suspicious_chunks_discarded", count=len(discarded))

            if not clean_chunks or clean_chunks[0]["similarity_score"] <= 0.0:
                emit("no_trusted_context")
                return {
                    "trace_id": tid,
                    "answer": OUT_OF_CONTEXT_FALLBACK,
                    "sources": [],
                    "telemetry": telemetry,
                }

            sources = [c["doc_id"] for c in clean_chunks]
            context_block = "\n\n".join(
                f"[{c['doc_id']}] {c['title']}: {c['content']}" for c in clean_chunks
            )

            answer_text, in_tok, out_tok = self._synthesize_rag_answer(
                question=question,
                context_block=context_block,
                clean_chunks=clean_chunks,
                sources=sources,
                use_live_api=use_live_api,
            )

            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            cost_usd = calculate_llm_cost_usd(in_tok, out_tok)
            emit(
                "llm_synthesis_completed",
                input_tokens=in_tok,
                output_tokens=out_tok,
                estimated_cost_usd=cost_usd,
                latency_ms=latency_ms,
            )

            return {
                "trace_id": tid,
                "answer": answer_text,
                "sources": sources,
                "telemetry": telemetry,
            }
        finally:
            trace_id_var.reset(token)

    def _synthesize_rag_answer(
        self,
        question: str,
        context_block: str,
        clean_chunks: List[Dict[str, Any]],
        sources: List[str],
        use_live_api: bool,
    ) -> Tuple[str, int, int]:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if use_live_api and api_key:
            try:
                return self._call_gemini_rest(question, context_block, sources, api_key)
            except Exception:
                # Graceful fallback to deterministic grounded synthesizer
                pass

        top_chunk = clean_chunks[0]
        second_chunk = clean_chunks[1] if len(clean_chunks) > 1 else top_chunk
        answer = (
            f"Based on CENABAST regulatory guidelines, {top_chunk['content']} "
            f"[source: {top_chunk['doc_id']}]. Furthermore, {second_chunk['content']} "
            f"[source: {second_chunk['doc_id']}]."
        )
        in_tok = estimate_tokens(question) + estimate_tokens(context_block)
        out_tok = estimate_tokens(answer)
        return answer, in_tok, out_tok

    @staticmethod
    def _call_gemini_rest(
        question: str,
        context_block: str,
        sources: List[str],
        api_key: str,
    ) -> Tuple[str, int, int]:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.5-flash:generateContent?key={api_key}"
        )
        payload = {
            "systemInstruction": {
                "parts": [
                    {
                        "text": (
                            "You are a senior CENABAST public healthcare procurement auditor. "
                            "Answer strictly using the provided regulatory context and always "
                            "include explicit inline citations in the format [source: <doc_id>]."
                        )
                    }
                ]
            },
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"Context:\n{context_block}\n\nQuestion: {question}"
                        }
                    ]
                }
            ],
        }
        data_bytes = json.dumps(payload).encode("utf-8")

        def _do_request() -> Dict[str, Any]:
            req = urllib.request.Request(
                url,
                data=data_bytes,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    classify_http_status(resp.status)
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as http_err:
                classify_http_status(http_err.code, str(http_err.reason))
                raise
            except urllib.error.URLError as url_err:
                raise TimeoutError(f"Network transport error: {url_err}") from url_err

        raw_resp = with_retry(_do_request, max_attempts=3, backoff_seconds=0.2)
        text = (
            raw_resp.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
            .strip()
        )
        if not text:
            raise RuntimeError("Empty response candidate from Gemini API.")
        if "[source:" not in text and sources:
            text = f"{text} [source: {sources[0]}]"

        usage = raw_resp.get("usageMetadata", {})
        in_tok = int(usage.get("promptTokenCount", estimate_tokens(context_block)))
        out_tok = int(usage.get("candidatesTokenCount", estimate_tokens(text)))
        return text, in_tok, out_tok

    def audit_tender_bid(
        self,
        active_ingredient: str,
        offered_unit_price: float,
        benchmark_median_price: float,
        fair_ceiling_p75_price: float,
        supplier_reliability_score: float = 85.0,
        proposed_share_pct: float = 40.0,
        currency: str = "USD",
    ) -> TenderAuditResult:
        """
        Evaluate a specific supplier tender bid against empirical IQR Fair-Price Bands
        and CENABAST regulatory rules, returning a Gate 1 + Gate 2 compliant
        `TenderAuditResult` with mandatory regulatory citations and HITL escalation flag.
        """
        if benchmark_median_price <= 0 or offered_unit_price < 0:
            raise ValueError("Prices must be positive.")

        token = trace_id_var.set(new_trace_id())
        tid = trace_id_var.get()
        telemetry: List[Dict[str, Any]] = []

        try:
            deviation_pct = round(
                ((offered_unit_price - benchmark_median_price) / benchmark_median_price) * 100.0,
                2,
            )
            overprice_detected = bool(
                offered_unit_price > fair_ceiling_p75_price or deviation_pct > 15.0
            )

            rag_query = (
                f"CENABAST price band audit for {active_ingredient} deviation {deviation_pct:.1f}% "
                f"supplier share {proposed_share_pct:.1f}% SLA reliability {supplier_reliability_score:.1f}"
            )
            retrieved = self.index.retrieve(rag_query, k=3)
            clean_chunks, _ = filter_suspicious_chunks(retrieved)
            citations = [c["doc_id"] for c in clean_chunks] or [
                "ley_19886_compras_publicas.md"
            ]

            if deviation_pct > 30.0 or supplier_reliability_score < 70.0 or proposed_share_pct > 45.0:
                recommendation = "reject"
            elif overprice_detected:
                recommendation = "negotiate"
            else:
                recommendation = "award"

            requires_hitl = bool(overprice_detected or recommendation in ("negotiate", "reject"))

            reasons: List[str] = []
            if overprice_detected:
                reasons.append(
                    f"Offered unit price ({offered_unit_price:,.4f} {currency}) deviates by "
                    f"{deviation_pct:+.1f}% from benchmark median ({benchmark_median_price:,.4f} {currency}) "
                    f"and breaches the P75 ceiling ({fair_ceiling_p75_price:,.4f} {currency}) "
                    f"[source: ley_19886_compras_publicas.md]."
                )
            else:
                reasons.append(
                    f"Offered unit price ({offered_unit_price:,.4f} {currency}, {deviation_pct:+.1f}% vs. median) "
                    f"is within the empirical P25-P75 fair-price band [source: catalogo_bandas_precios_cenabast_2026.md]."
                )

            if proposed_share_pct > 45.0:
                reasons.append(
                    f"Proposed supplier allocation ({proposed_share_pct:.1f}%) violates the <= 45% "
                    f"anti-monopoly multi-sourcing cap [source: directiva_cenabast_multioferente_2026.md]."
                )
            if supplier_reliability_score < 70.0:
                reasons.append(
                    f"Supplier SLA score ({supplier_reliability_score:.1f}/100) falls below the 70.0 "
                    f"minimum hospital delivery threshold [source: manual_sla_logistica_hospitalaria.md]."
                )

            ruling_text = " ".join(reasons)

            telemetry.append(
                log_telemetry_event(
                    "tender_bid_audited",
                    deviation_pct=deviation_pct,
                    overprice_detected=overprice_detected,
                    recommendation=recommendation,
                    requires_hitl=requires_hitl,
                    citations_count=len(citations),
                )
            )

            return TenderAuditResult(
                trace_id=tid,
                active_ingredient=active_ingredient,
                offered_unit_price=offered_unit_price,
                benchmark_median_price=benchmark_median_price,
                fair_ceiling_p75_price=fair_ceiling_p75_price,
                currency=currency,
                overprice_detected=overprice_detected,
                deviation_pct=deviation_pct,
                recommendation=recommendation,
                ruling=ruling_text,
                citations=citations,
                requires_human_supervision=requires_hitl,
                telemetry_events=telemetry,
            )
        finally:
            trace_id_var.reset(token)
