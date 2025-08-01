"""
Retrieval Pipeline Scaffolding

Stages:
1) Query diversification
2) Multi-engine search adapters
3) Authority scoring and filtering (whitelists/blacklists)
4) Fetch and parse (HTML/PDF/text) with metadata
5) Domain-aware chunking
6) Embeddings generation
7) Deduplication (MD5, SimHash)
8) Persistence hooks (store source/chunk/embedding/metadata)

Notes:
- This scaffolding avoids concrete provider code; wire to existing MCP servers
  or HTTP clients.
- Embedding provider and parser/HTTP fetcher are passed as interfaces for easy
  substitution.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


# ----------------------------
# Protocols (Interfaces)
# ----------------------------

class SearchAdapter(Protocol):
    def search(
        self,
        query: str,
        *,
        timeframe: Optional[str] = None,
        site: Optional[str] = None,
        num: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Return a list of results with minimally:
        { "url": str, "title": str, "snippet": str, "rank": int, "raw": dict }
        """


class HttpFetcher(Protocol):
    def fetch(self, url: str, *, timeout: int = 15) -> Tuple[bytes, str]:
        """
        Fetch content at url, return (content_bytes, mime_type).
        """


class Parser(Protocol):
    def parse(self, content: bytes, mime_type: str) -> Dict[str, Any]:
        """
        Return:
          {
            "text": str,
            "title": Optional[str],
            "sections": Optional[List[dict]],
            "metadata": dict
          }
        """


class Chunker(Protocol):
    def chunk(self, doc: "ParsedDocument") -> List["Chunk"]:
        """
        Create domain-aware chunks from parsed document.
        """
        raise NotImplementedError("Chunker.chunk must be implemented by a concrete class")


class Embedder(Protocol):
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Return embeddings for provided texts.
        """


class Persistence(Protocol):
    def upsert_source(self, source: "Source") -> None: ...
    def upsert_chunks(self, chunks: List["Chunk"]) -> None: ...
    def upsert_embeddings(self, embeddings: List["Embedding"]) -> None: ...


# ----------------------------
# Data classes
# ----------------------------

@dataclass
class SearchResult:
    url: str
    title: Optional[str]
    snippet: Optional[str]
    rank: int
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    url: str
    title: Optional[str]
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Source:
    id: str
    url: str
    title: Optional[str]
    tld: Optional[str]
    published_at: Optional[str] = None
    updated_at: Optional[str] = None
    fetched_at: Optional[str] = None
    parser: Optional[str] = None
    mime_type: Optional[str] = None
    authority_scores: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    source_id: str
    section: Optional[str]
    position: int
    text: str
    token_count: Optional[int]
    md5_hash: str
    simhash: Optional[int]
    credibility_features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Embedding:
    id: str
    chunk_id: str
    model: str
    dim: int
    vector: List[float]


# ----------------------------
# Utilities
# ----------------------------

def md5_hex(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def simhash64(tokens: Iterable[str]) -> int:
    """
    Simple 64-bit simhash from tokens.
    This is a placeholder; replace with a robust implementation as needed.
    Falls back gracefully if mmh3 is unavailable.
    """
    from collections import Counter
    try:
        import mmh3  # type: ignore
        use_mmh3 = True
    except Exception:
        use_mmh3 = False

    weights = Counter(tokens)
    bit_counts = [0] * 64
    for token, w in weights.items():
        if use_mmh3:
            h = mmh3.hash64(token, signed=False)[0]
        else:
            # Fallback: use Python's hash, normalized to 64-bit unsigned
            h = (hash(token) & ((1 << 64) - 1))
        for i in range(64):
            bit = 1 if (h >> i) & 1 else -1
            bit_counts[i] += w * bit
    fingerprint = 0
    for i in range(64):
        if bit_counts[i] >= 0:
            fingerprint |= (1 << i)
    return fingerprint


def estimate_tokens(text: str) -> int:
    # Rough heuristic; replace with tiktoken if desired.
    words = max(1, len(text.split()))
    # approx 0.75 words per token -> tokens ~ words / 0.75
    return int(round(words / 0.75))


def extract_tld(url: str) -> Optional[str]:
    try:
        from urllib.parse import urlparse
        host = urlparse(url).hostname or ""
        return ".".join(host.split(".")[-2:]) if host else None
    except Exception:
        return None


# ----------------------------
# Query Diversification
# ----------------------------

@dataclass
class DiversificationConfig:
    site_whitelist: List[str] = field(default_factory=lambda: [])
    site_blacklist: List[str] = field(default_factory=lambda: [])
    timeframes: List[str] = field(
        default_factory=lambda: ["past year", "past 2 years"]
    )
    reformulations: int = 3
    max_queries: int = 12


def diversify_queries(
    seed_query: str,
    cfg: DiversificationConfig,
) -> List[Dict[str, Optional[str]]]:
    """
    Produce diversified query variants:
    - raw
    - reformulated
    - site/time constrained
    """
    variants: List[Dict[str, Optional[str]]] = []
    # base
    variants.append({"q": seed_query, "timeframe": None, "site": None})

    # simple reformulations (placeholder; replace with LLM-based reformulation)
    stems = [
        f"{seed_query}",
        f"{seed_query} literature review",
        f"{seed_query} systematic review methods",
        f"{seed_query} latest findings",
        f"{seed_query} evidence and citations",
    ][: cfg.reformulations + 1]

    # site constraints
    site_targets = (
        cfg.site_whitelist or [None, "site:.gov", "site:.edu", "site:.org"]
    )

    for stem in stems:
        for tf in (cfg.timeframes or [None]):
            for site in site_targets:
                variants.append({"q": stem, "timeframe": tf, "site": site})

    # blacklist filter (exclude sites matching blacklist)
    def blocked(v: Dict[str, Optional[str]]) -> bool:
        site = v.get("site") or ""
        return any(b for b in cfg.site_blacklist if b and b in site)

    pruned = [v for v in variants if not blocked(v)]

    # cap
    return pruned[: cfg.max_queries]


# ----------------------------
# Authority Scoring
# ----------------------------

@dataclass
class AuthorityConfig:
    tld_bonus: Dict[str, float] = field(
        default_factory=lambda: {
            ".gov": 0.3,
            ".edu": 0.25,
            ".org": 0.1,
        }
    )
    recency_decay_half_life_days: int = 365
    whitelist_bonus: float = 0.2
    blacklist_penalty: float = -0.5


def authority_score(
    url: str,
    *,
    tld: Optional[str],
    cfg: AuthorityConfig,
    whitelisted: bool,
    blacklisted: bool,
) -> float:
    score = 0.0
    if tld:
        for suffix, bonus in cfg.tld_bonus.items():
            if tld.endswith(suffix):
                score += bonus
    if whitelisted:
        score += cfg.whitelist_bonus
    if blacklisted:
        score += cfg.blacklist_penalty
    return max(0.0, score)


# ----------------------------
# Pipeline
# ----------------------------

@dataclass
class RetrievalConfig:
    top_k_results: int = 20
    max_fetch: int = 40
    chunk_max_chars: int = 1800
    chunk_overlap_chars: int = 200
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3072
    whitelist_domains: List[str] = field(default_factory=lambda: [])
    blacklist_domains: List[str] = field(default_factory=lambda: [])


class SimpleHtmlChunker(Chunker):
    def __init__(self, max_chars: int = 1800, overlap: int = 200) -> None:
        self.max_chars = max_chars
        self.overlap = overlap

    def chunk(self, doc: ParsedDocument) -> List[Chunk]:
        text = doc.text or ""
        chunks: List[Chunk] = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(len(text), start + self.max_chars)
            piece = text[start:end]
            md5 = md5_hex(piece)
            sim = simhash64(piece.split())
            chunks.append(
                Chunk(
                    id=md5,  # scaffold: use md5 as stable id for now
                    source_id=md5_hex(doc.url),
                    section=None,
                    position=idx,
                    text=piece,
                    token_count=estimate_tokens(piece),
                    md5_hash=md5,
                    simhash=sim,
                )
            )
            idx += 1
            start = end - self.overlap
            if start <= 0:
                start = end
        return chunks


# ----------------------------
# Credibility Card and Critic/Self-check
# ----------------------------

@dataclass
class CredibilityCard:
    source_id: str
    url: str
    tld: Optional[str]
    features: Dict[str, float]
    score: float


def generate_credibility_card(
    *,
    url: str,
    source_id: str,
    tld: Optional[str],
    chunks: List[Chunk],
    authority_cfg: Optional[AuthorityConfig] = None,
    whitelist: Optional[List[str]] = None,
    blacklist: Optional[List[str]] = None,
) -> CredibilityCard:
    """
    Compute a simple credibility score with feature breakdown:
    - domain reputation (via tld bonus)
    - recency proxy (not available here; set to 0.0 placeholder)
    - citation density proxy (count of bracket-like tokens per 1k tokens)
    - cross-source agreement proxy (placeholder 0.0)
    """
    authority_cfg = authority_cfg or AuthorityConfig()
    whitelist = whitelist or []
    blacklist = blacklist or []

    is_white = bool(tld and any(w for w in whitelist if w and tld.endswith(w)))
    is_black = bool(tld and any(b for b in blacklist if b and tld.endswith(b)))
    dom_rep = authority_score(
        url,
        tld=tld,
        cfg=authority_cfg,
        whitelisted=is_white,
        blacklisted=is_black,
    )

    # citation density proxy: count brackets patterns in text
    total_tokens = max(1, sum((c.token_count or 0) for c in chunks))
    bracket_hits = 0
    for c in chunks:
        txt = c.text
        bracket_hits += txt.count("[") + txt.count("]") + txt.count("(") + txt.count(")")
    citation_density = min(1.0, (bracket_hits / max(1, total_tokens)) * 1000.0)

    recency = 0.0  # placeholder until published/updated timestamps are integrated
    agreement = 0.0  # placeholder until cross-source agreement is computed

    # Weighted sum
    features = {
        "domain_reputation": float(dom_rep),
        "recency": float(recency),
        "citation_density": float(citation_density),
        "cross_source_agreement": float(agreement),
    }
    score = max(
        0.0,
        min(
            1.0,
            0.6 * features["domain_reputation"]
            + 0.1 * features["recency"]
            + 0.2 * features["citation_density"]
            + 0.1 * features["cross_source_agreement"],
        ),
    )
    return CredibilityCard(
        source_id=source_id,
        url=url,
        tld=tld,
        features=features,
        score=score,
    )


def run_critic_self_check(
    *,
    claims: List[str],
    claim_to_chunks: Dict[int, List[Chunk]],
    min_citation_density: float = 0.02,
) -> Dict[str, Any]:
    """
    Enforce simple pre-finalization checks:
    - Each claim must have at least one evidence chunk
    - Per-claim citation density proxy must exceed a threshold
    - Return issues list for remediation
    """
    issues: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []

    for idx, claim in enumerate(claims):
        chunks = claim_to_chunks.get(idx, [])
        if not chunks:
            issues.append(
                {
                    "type": "missing_evidence",
                    "claim_index": idx,
                    "message": "No evidence chunks linked to claim",
                }
            )
            results.append(
                {
                    "claim_index": idx,
                    "ok": False,
                    "citation_density": 0.0,
                    "evidence_count": 0,
                }
            )
            continue

        total_tokens = max(1, sum((c.token_count or 0) for c in chunks))
        bracket_hits = 0
        for c in chunks:
            txt = c.text
            bracket_hits += txt.count("[") + txt.count("]") + txt.count("(") + txt.count(")")
        density = (bracket_hits / total_tokens) if total_tokens else 0.0

        ok = density >= min_citation_density
        if not ok:
            issues.append(
                {
                    "type": "low_citation_density",
                    "claim_index": idx,
                    "message": f"Citation density {density:.4f} below threshold {min_citation_density:.4f}",
                }
            )

        results.append(
            {
                "claim_index": idx,
                "ok": ok,
                "citation_density": float(density),
                "evidence_count": len(chunks),
            }
        )

    return {
        "results": results,
        "issues": issues,
        "passed": len(issues) == 0,
    }


class RetrievalPipeline:
    def __init__(
        self,
        adapters: List[SearchAdapter],
        fetcher: HttpFetcher,
        parser: Parser,
        chunker: Chunker,
        embedder: Embedder,
        persistence: Persistence,
        div_cfg: Optional[DiversificationConfig] = None,
        auth_cfg: Optional[AuthorityConfig] = None,
        r_cfg: Optional[RetrievalConfig] = None,
    ) -> None:
        self.adapters = adapters
        self.fetcher = fetcher
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.persistence = persistence
        self.div_cfg = div_cfg or DiversificationConfig()
        self.auth_cfg = auth_cfg or AuthorityConfig()
        self.r_cfg = r_cfg or RetrievalConfig()

    def run(self, seed_query: str) -> Dict[str, Any]:
        t0 = time.time()

        # 1) Diversify
        q_variants = diversify_queries(seed_query, self.div_cfg)
        logger.info("Diversified into %s queries", len(q_variants))

        # 2) Multi-adapter search
        dedup_urls: Dict[str, SearchResult] = {}
        for adapter in self.adapters:
            for qv in q_variants:
                qtext = qv.get("q") or ""
                results = adapter.search(
                    qtext,
                    timeframe=qv.get("timeframe"),
                    site=qv.get("site"),
                    num=self.r_cfg.top_k_results,
                )
                for r in results:
                    url = r.get("url")
                    if not url or url in dedup_urls:
                        continue
                    dedup_urls[url] = SearchResult(
                        url=url,
                        title=r.get("title"),
                        snippet=r.get("snippet"),
                        rank=r.get("rank", 0),
                        raw=r,
                    )
        logger.info("Collected %s unique URLs", len(dedup_urls))

        # 3) Authority scoring + domain filters
        ranked: List[Tuple[float, SearchResult]] = []
        for url, sr in dedup_urls.items():
            tld = extract_tld(url)
            whitelisted = any(
                d for d in self.r_cfg.whitelist_domains if d and d in (tld or "")
            )
            blacklisted = any(
                d for d in self.r_cfg.blacklist_domains if d and d in (tld or "")
            )
            score = authority_score(
                url,
                tld=tld,
                cfg=self.auth_cfg,
                whitelisted=whitelisted,
                blacklisted=blacklisted,
            )
            ranked.append((score, sr))
        ranked.sort(key=lambda x: x[0], reverse=True)

        # 4) Fetch and parse
        fetched_docs: List[ParsedDocument] = []
        for _, sr in ranked[: self.r_cfg.max_fetch]:
            try:
                content, mime = self.fetcher.fetch(sr.url)
                parsed = self.parser.parse(content, mime)
                doc = ParsedDocument(
                    url=sr.url,
                    title=parsed.get("title") or sr.title,
                    text=parsed.get("text") or "",
                    metadata={"mime_type": mime, **(parsed.get("metadata") or {})},
                )
                fetched_docs.append(doc)

                # Persist source
                source = Source(
                    id=md5_hex(sr.url),
                    url=sr.url,
                    title=doc.title,
                    tld=extract_tld(sr.url),
                    parser=self.parser.__class__.__name__,
                    mime_type=mime,
                    authority_scores={"authority": float(next((sc for sc, srr in ranked if srr.url == sr.url), 0.0))},
                    metadata={"rank": sr.rank},
                )
                self.persistence.upsert_source(source)
            except Exception as e:
                logger.warning("Fetch/parse failed for %s: %s", sr.url, e)

        # 5) Chunk
        all_chunks: List[Chunk] = []
        for doc in fetched_docs:
            try:
                chunks = self.chunker.chunk(doc)
                # Dedup exact by MD5 (in-memory filter here; persistence layer should also enforce)
                seen_md5 = set()
                unique_chunks = []
                for ch in chunks:
                    if ch.md5_hash in seen_md5:
                        continue
                    seen_md5.add(ch.md5_hash)
                    unique_chunks.append(ch)
                all_chunks.extend(unique_chunks)
                self.persistence.upsert_chunks(unique_chunks)
            except Exception as e:
                logger.warning("Chunking failed for %s: %s", doc.url, e)

        # 6) Embeddings
        texts = [c.text for c in all_chunks]
        vectors = self.embedder.embed(texts) if texts else []
        emb_records: List[Embedding] = []
        for c, v in zip(all_chunks, vectors):
            emb_records.append(
                Embedding(
                    id=f"{c.id}:{self.r_cfg.embedding_model}",
                    chunk_id=c.id,
                    model=self.r_cfg.embedding_model,
                    dim=len(v),
                    vector=v,
                )
            )
        if emb_records:
            self.persistence.upsert_embeddings(emb_records)

        t1 = time.time()
        return {
            "query": seed_query,
            "queries_emitted": len(q_variants),
            "urls_considered": len(dedup_urls),
            "docs_parsed": len(fetched_docs),
            "chunks_indexed": len(all_chunks),
            "embeddings_indexed": len(emb_records),
            "elapsed_sec": round(t1 - t0, 3),
        }