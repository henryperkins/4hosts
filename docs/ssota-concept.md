# SSOTA Concept Overview

SSOTA is a concise, production‑minded synthesis of the Four Hosts agentic research materials in this folder. It preserves the core paradigm model (Dolores/Teddy/Bernard/Maeve), the W‑S‑C‑I context engineering pipeline, and the end‑to‑end flow from classification to synthesis. This document defines the concept and principles, with pointers for implementation and links to deeper guides.

---

## Purpose

- Map each research query to the most effective paradigm.
- Engineer context via Write → Select → Compress → Isolate (W‑S‑C‑I).
- Execute paradigm‑aware research and synthesize answers with citations.
- Remain transparent, adaptable (self‑healing), and ethical.

## Core Idea

1) Classify a query into a probability distribution over four paradigms, choose a primary (and optional secondary).  
2) Pass the classification through W‑S‑C‑I to generate engineered search inputs and extraction rules.  
3) Run paradigm‑aware search + credibility scoring, dedupe, and fact extraction.  
4) Synthesize an answer in the paradigm’s voice/structure; optionally integrate a secondary paradigm (mesh).

## Scope and Non‑Goals

- Scope: paradigm classification, W‑S‑C‑I context engineering, paradigm‑aware research, and synthesis with citations.  
- Non‑Goals: vendor‑specific commitments, full compliance text, or finalized pricing; those live in product/ops docs outside this folder.

## The Four Paradigms (snapshots)

- Dolores (Revolutionary): expose systemic injustices and power imbalances; emphasize patterns of oppression and calls to action.
- Teddy (Devotion): protect and support vulnerable communities; emphasize needs, resources, and humane care.
- Bernard (Analytical): seek empirical truth; emphasize data, causal analysis, limitations, and research directions.
- Maeve (Strategic): gain advantage and control; emphasize tactics, implementation steps, and success metrics.

## W‑S‑C‑I in One Page

- Write: derive documentation focus, themes, narrative frame, and search priorities from the paradigm + query features.
- Select: generate enhanced queries, source preferences, exclusion filters, and tool selections; include secondary‑paradigm probes when signal is strong.
- Compress: set paradigm‑specific compression ratio and priority/removed elements; compute a token budget based on query complexity.
- Isolate: define extraction patterns, key‑finding criteria, focus areas, and output schema aligned to the paradigm.

## Context Engineering (applied)

- Context window as RAM: treat prompt tokens as scarce working memory; W‑S‑C‑I exists to curate exactly what enters the window each step.  
- Context types to manage:  
  - Instructions (system prompts, tool descriptions, few‑shot exemplars).  
  - Knowledge (retrieved facts, prior findings, notes).  
  - Tools (feedback from web search, credibility scoring, extraction).  
- Memory separation: keep most artifacts outside the context window (structured state stores) and inject only what is needed per step.  
- Failure modes to avoid: context poisoning (bad content enters memory), distraction (too much), confusion (irrelevant), clash (contradictions). Use filters, dedupe, conflict flags, and summarization to mitigate.

## Guiding Principles

- Paradigm transparency: always show which paradigm is active and why.
- Source diversity: prefer multiple high‑quality, paradigm‑appropriate sources; flag bias and uncertainty.
- User agency: allow overrides and explain trade‑offs; support paradigm switching (self‑healing) when progress stalls.
- Ethics: avoid sensationalism; cite clearly; respect privacy; do not fabricate statistics.
 - Context hygiene: trim aggressively, isolate risky tool outputs, and prefer verified content for injection.

## Assumptions & Constraints

- Metrics labeled as targets or prototype benchmarks exclude external web I/O unless noted.  
- Examples are illustrative; when numbers appear, they must be sourced or clearly marked as placeholders.  
- Westworld character names are metaphors for paradigms; no affiliation is implied.

## Example (summarized)

“How can small businesses compete with Amazon?” → Primary: Maeve (Strategic), Secondary: Dolores (anti‑monopoly context).

- Write: map competitive landscape and local advantages; note anti‑monopoly angles.
- Select: generate strategy/tactics queries and prioritize industry/case‑study sources; add antitrust probes.
- Compress: keep ~40% focusing on actions/metrics.
- Isolate: extract strategic opportunities, implementation steps, and success metrics.

## Targets (from concept materials)

- Classification accuracy: ≥85% (hybrid rule + LLM).  
- End‑to‑end processing: sub‑second for classification + W‑S‑C‑I (excluding external web I/O).  
- Answer quality: high relevance with correct, visible citations.  
- Cost/time controls: caching for classifications and search; adjustable depth.

## Decision Flow (high level)

- Classify → Engineer context (W‑S‑C‑I) → Search + analyze → Synthesize → Display paradigm + citations → Optional self‑healing switch or mesh integration.

## Glossary

- W‑S‑C‑I: Write, Select, Compress, Isolate context engineering pipeline.
- Self‑healing: automatic paradigm switch when the current path underperforms (e.g., analysis paralysis).
- Mesh network: optional aggregation of complementary paradigms for complex queries.
- Paradigm distribution: probability over {Dolores, Teddy, Bernard, Maeve} with primary/secondary selection.

---
Cross‑references: see ssota-architecture.md (system specifics), ssota-context-and-prompt-engineering.md (W‑S‑C‑I and prompts), and ssota-phased-implementation.md (delivery plan).

This SSOTA concept consolidates and normalizes the intent and practices described across: Architecture, Classification & Context System, System Integration, API v1, MVP, Phased Plan, Risk Matrix, and Visual Flow guides in this folder.

Version: v1 (compiled from concept folder), Date: 2025‑08‑30
