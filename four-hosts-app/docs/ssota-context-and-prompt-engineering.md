# SSOTA Context and Prompt Engineering

This guide unifies the W‑S‑C‑I context pipeline and practical prompt patterns, derived entirely from the concept docs here. Use it to drive consistent LLM behavior across paradigms.

---

## W‑S‑C‑I: Responsibilities and Artifacts

- Write → outputs focus, themes, narrative frame, search priorities.  
- Select → outputs enhanced queries, source preferences, exclusion filters, tools, and max sources; may add secondary‑paradigm probes.  
- Compress → outputs ratio, strategy, priority vs removed elements, token budget from complexity.  
- Isolate → outputs extraction strategy, criteria, regex patterns, focus areas, and a structured findings schema.

I/O convention: each layer returns a compact JSON object with only the parameters needed by downstream components; avoid mixing extracted content with configuration. Keep heavy artifacts in memory stores and inject only minimal snippets.

### Context Inputs Layering

- Instructions: system role + paradigm system prompt + minimal tool descriptions.  
- Knowledge: retrieved snippets (post‑compression) and prior findings directly relevant to the current step.  
- Tools: specific, sanitized tool outputs needed for the step, with identifiers to cite later.  
- Assembly order: instructions → task directive → minimal knowledge/tool snippets → output schema/format spec.

## Paradigm Config Snapshots (authoritative)

### Dolores (Revolutionary)
- Write: document injustices, power imbalances; themes: oppression/resistance/truth/justice.
- Select: add modifiers like expose/reveal/uncover; prioritize investigative/alternative sources; exclude corporate PR.
- Compress: ~0.7; preserve impact; prioritize evidence of wrongdoing, testimonies, power dynamics.
- Isolate: strategy pattern_of_injustice; criteria: systemic patterns, power imbalances, victim impact, accountability gaps.

### Teddy (Devotion)
- Write: empathetic profiles, care needs; themes: compassion/support/healing/community.
- Select: add support/help/resources; prioritize community/nonprofit/educational; exclude exploitative/sensational.
- Compress: ~0.6; preserve human stories; prioritize resources and positive outcomes.
- Isolate: strategy human_centered_needs; criteria: individual needs, support, success stories, care strategies.

### Bernard (Analytical)
- Write: systematic documentation; themes: analysis/causation/methodology/validation.
- Select: add research/study/analysis/data; prioritize peer‑reviewed/government/statistical; exclude unverified/opinion.
- Compress: ~0.5; data distillation; prioritize statistical findings and causality; remove anecdotes.
- Isolate: strategy empirical_extraction; criteria: significance, causal links, gaps, methodological strengths.

### Maeve (Strategic)
- Write: map strategic landscape and actionable opportunities; themes: leverage/advantage/optimization.
- Select: add strategy/tactics/optimize/leverage; prioritize industry/consultancy/case studies.
- Compress: ~0.4; action extraction; prioritize tactics, steps, metrics; remove background theory.
- Isolate: strategy strategic_intelligence; criteria: competitive advantages, tactics, success metrics, resources.

---

## Prompt Patterns

Note: Patterns below are distilled from the documented behaviors. Adapt variable names to your orchestration layer. Keep citations separate from reasoning.

### 1) Classification (rule/LLM hybrid)

Instruction skeleton:

```
You map a single query to four paradigms: dolores (revolutionary), teddy (devotion), bernard (analytical), maeve (strategic).
Analyze tokens, entities, intent (how_to/why/what), domain hints, urgency, complexity, and emotional valence.
Return a normalized distribution, primary, optional secondary (if >0.2), and brief reasoning per paradigm.
Do not fabricate facts; do not cite specific external sources here.
```

Expected JSON shape (see ClassificationResult in architecture guide).

Minimum output fields:

```json
{
  "primary": "maeve",
  "secondary": "dolores",
  "distribution": {"maeve": 0.40, "dolores": 0.25, "bernard": 0.20, "teddy": 0.15},
  "confidence": 0.78,
  "reasoning": {"maeve": ["..."], "dolores": ["..."]}
}
```

### 2) Write Layer

Per‑paradigm focus prompt:

```
Given the query and its classification, produce:
- documentation_focus (single sentence in paradigm’s voice)
- key_themes (≤10)
- narrative_frame (short phrase)
- search_priorities (≤8)
Follow the paradigm’s strategy strictly; do not invent statistics.
```

Output shape:

```json
{
  "documentation_focus": "...",
  "key_themes": ["..."],
  "narrative_frame": "...",
  "search_priorities": ["..."]
}
```

Context injection: select only the top‑ranked items within budget (e.g., ≤5 themes, ≤5 priorities) before adding to prompts.

### 3) Select Layer

Query expansion prompt:

```
Generate up to 10 enhanced search queries:
- Include: original, paradigm_modified (≤3), theme_enhanced (≤2), entity_focused (≤2).
- Provide type and weight (1.0 original; 0.8 modified; 0.7 theme; 0.6 entity).
- Provide source_preferences, exclusion_filters, tool_selections from the paradigm config.
If secondary paradigm weight >0.25, add up to 3 secondary_paradigm queries.
```

Output shape:

```json
{
  "queries": [
    {"query": "...", "type": "original", "weight": 1.0},
    {"query": "... strategy", "type": "paradigm_modified", "weight": 0.8}
  ],
  "source_preferences": ["industry"],
  "exclusion_filters": ["opinion"],
  "tool_selections": ["market_analysis"],
  "max_sources": 100
}
```

Pre‑send checks:
- Remove near‑duplicate queries; cap total (≤10).  
- Ensure entity_focused queries are well‑formed (quoted entities).  
- Bind source_preferences to credibility policies of the active paradigm.

### 4) Compress Layer

Compression instruction:

```
Set compression_ratio and strategy per paradigm. Choose priority_elements to retain and removed_elements to drop.
Compute token_budget as base_tokens * (1 + complexity_score) * ratio.
Do not summarize sources here; only return parameters.
```

Output shape:

```json
{
  "compression_ratio": 0.4,
  "compression_strategy": "action_extraction",
  "priority_elements": ["specific tactics", "success metrics"],
  "removed_elements": ["background theory"],
  "token_budget": 1300
}
```

Compression tactics:
- Summarize long tool outputs into structured bullet points; strip boilerplate.  
- Prefer extracting facts with identifiers to enable later citation mapping.  
- Enforce per‑layer token caps; drop low‑priority elements first.

### 5) Isolate Layer

Extraction setup:

```
Define isolation_strategy, key_findings_criteria, regex extraction_patterns, focus_areas, and a structured output schema tailored to the paradigm.
Do not extract yet; only return the schema and criteria.
```

Output shape:

```json
{
  "isolation_strategy": "strategic_intelligence",
  "key_findings_criteria": ["competitive advantages", "implementation tactics"],
  "extraction_patterns": ["strategy\\s+to\\s+\\w+"],
  "focus_areas": ["quick wins", "long-term positioning"],
  "output_structure": {"strategic_opportunities": [], "implementation_steps": [], "success_metrics": []}
}
```

Isolation tactics:
- Keep raw tool outputs quarantined until referenced by a specific extraction pattern.  
- For risky content (unverified or sensational), require a second independent source before inclusion.

### 6) Answer Synthesis

Per‑paradigm templates (from concept docs):

- Dolores: Opening (injustice) → Evidence chains → Context (systemic/historical) → Action → Resources.
- Teddy: Opening (empathetic) → Understanding needs → Support resources → Best practices → Next steps.
- Bernard: Opening (objective summary) → Analysis (data/patterns/causality) → Limitations → Further research.
- Maeve: Opening (strategic overview) → Opportunities → Implementation framework → Metrics → Timeline.

Synthesis instruction:

```
Use the paradigm’s template and the extracted findings to produce a structured answer. Keep claims tied to sources; include citations as separate objects. If secondary paradigm exists, add an Additional Perspective section.
```

Citation shape (example):

```json
{ "id": "cite_001", "title": "...", "source": "...", "url": "...", "credibility_score": 0.9, "paradigm_alignment": 0.9 }
```

Prompt assembly (pseudo):

```
SYSTEM: <paradigm system prompt>
TOOLS: <minimal tool descriptions>
TASK: Synthesize answer using the provided findings; adhere to the template.
CONTEXT:
- Findings (structured): <isolated findings JSON>
- Snippets (compressed): <bullet points>
OUTPUT: Return structured sections + citations as JSON per schema.
```

---

## Self‑Healing and Mesh Prompts

Self‑healing (switch triggers):

```
If no high‑quality findings after N searches or answer quality < threshold, suggest a paradigm switch:
- From Bernard → Maeve when analysis paralysis is detected.
- From Maeve → Bernard when evidence depth is insufficient.
- From any → Teddy when human needs/ethics dominate the query.
- From any → Dolores when systemic injustice patterns dominate.
Return: reason, from_paradigm, to_paradigm, confidence.
```

Mesh (integration):

```
If complementary insights exist across paradigms, produce a synthesis object:
- primary_narrative, supporting_perspectives, conflicts, unified_view.
```

---

## Guardrails

- No fabricated statistics or unverifiable claims; mark examples as illustrative.
- Keep paradigm transparency; show why a choice was made.
- Separate reasoning from citations; cite sources used in synthesis, not internal heuristics.
- Respect privacy; avoid echoing sensitive data from inputs.

Prompt hygiene:

- Place behavioral constraints in system messages; keep user prompts focused on content.  
- Separate style (paradigm tone) from facts.  
- Enforce JSON mode where structured outputs are required.  
- Escape regex patterns correctly (double‑escape in JSON strings).

Context failure modes to detect:
- Poisoning: do not carry forward hallucinated or unsourced claims; require verification.  
- Distraction: prune long irrelevant history; prefer summaries.  
- Confusion: avoid mixing unrelated topics in one prompt; split steps.  
- Clash: surface contradictions explicitly; do not silently merge.

This guide composes the paradigm strategies, W‑S‑C‑I layer responsibilities, and answer templates documented across the concept materials into reusable prompt and context patterns.
