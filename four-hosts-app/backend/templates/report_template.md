# Research Report

## Executive Summary
- Objective: <fill from intake.objective>
- Key Findings: <brief bullet list>
- Confidence/Uncertainty: <summary>
- Recommendations: <actions>

## Methods
- Scope and Constraints: <from intake.scope/constraints>
- Data Sources and Discovery:
  - Search engines/APIs used
  - Query strategies and time windows
  - Inclusion/exclusion criteria
- Retrieval and Processing:
  - Parsers, chunking strategy, embeddings model
  - Deduplication approach (MD5/SimHash)
  - Metadata collected
- Credibility Assessment:
  - Features: domain reputation, recency, citations, cross-source agreement
  - Scoring function and thresholds
- Orchestration:
  - Budget caps: tokens/cost/time
  - Tools and rate limits
  - Failure handling/backoff
- Evaluation:
  - Rubric criteria and weighting
  - Evaluator isolation and challenge sets

## Results
- Thematic Findings with Per-sentence Citations:
  1. Claim A. [CIT:source_id:chunk_id]
  2. Claim B. [CIT:source_id:chunk_id]
- Evidence Summaries:
  - Source S1 (credibility: X): <summary> [CIT:S1:Ck]
  - Source S2 (credibility: Y): <summary> [CIT:S2:Cm]
- Conflict Analysis:
  - Conflicting claims and adjudication notes
- Figures and Tables
  - Figure 1: <title> (provenance: <artifact_id or URL>)
  - Table 1: <title> (provenance: <artifact_id or dataset version>)

## Discussion
- Interpretation of findings
- Limitations and potential biases
- Uncertainty quantification per claim
- Implications and alternatives

## Recommendations
- Actionable next steps with priority and rationale
- Risks and mitigations

## References
- Automatic bibliography (ordered by first citation)
  - [S1] Author, Title, Venue, Year, URL, Accessed date.
  - [S2] ...

## Appendix
- Intake Scope Card snapshot
- Tool usage and costs
- Reproduction guide (env, seeds, versions)
- Full lineage and artifact IDs